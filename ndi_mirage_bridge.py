"""
NDI <-> Decart Mirage Bridge

Receives an NDI video feed, processes it through Decart's Mirage realtime
style-transfer API, and outputs the transformed frames back via NDI.

Requires:
  - NDI Runtime installed (https://ndi.video/tools/)
  - pip install decart[realtime] cyndilib numpy opencv-python av

Usage:
  python ndi_mirage_bridge.py --api-key YOUR_KEY
  python ndi_mirage_bridge.py --source "MY-PC (OBS)" --prompt "Cyberpunk city"
"""

import argparse
import asyncio
import fractions
import logging
import os
import queue
import signal
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from aiortc import MediaStreamTrack, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.sender import Sender
from cyndilib.video_frame import VideoFrameSync, VideoSendFrame
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.wrapper.ndi_structs import FourCC

from decart import DecartClient, models
from decart.realtime import RealtimeClient, RealtimeConnectOptions
from decart.types import ModelState, Prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECART_MODEL = "mirage_v2"
TARGET_FPS = 22
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704


# ---------------------------------------------------------------------------
# FrameBuffer: thread-safe latest-frame buffer
# ---------------------------------------------------------------------------


class FrameBuffer:
    """Thread-safe single-slot buffer holding the most recent NDI frame.

    The NDI receiver thread overwrites the latest frame; the async
    NDIVideoTrack reads it. Intermediate frames are silently dropped,
    which is fine since the Decart model only needs 22 fps and the NDI
    source may produce 30/60 fps.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None  # (H, W, 4) BGRX uint8
        self._new_frame = threading.Event()
        # FPS tracking
        self._fps_lock = threading.Lock()
        self._fps_count = 0
        self._fps_last_time = time.time()
        self._fps_value = 0.0

    def put(self, frame: np.ndarray) -> None:
        """Store a frame (called from NDI receiver thread)."""
        with self._lock:
            self._frame = frame
        self._new_frame.set()
        self._tick_fps()

    def get(self) -> Optional[np.ndarray]:
        """Return the latest frame or None (non-blocking)."""
        with self._lock:
            return self._frame

    def wait_for_first_frame(self, timeout: float = 30.0) -> bool:
        """Block until the first frame arrives."""
        return self._new_frame.wait(timeout)

    def _tick_fps(self) -> None:
        with self._fps_lock:
            self._fps_count += 1
            now = time.time()
            elapsed = now - self._fps_last_time
            if elapsed >= 1.0:
                self._fps_value = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_last_time = now

    @property
    def fps(self) -> float:
        with self._fps_lock:
            return self._fps_value


# ---------------------------------------------------------------------------
# NDI Receiver
# ---------------------------------------------------------------------------


class NDIReceiver:
    """Captures frames from an NDI source in a background thread."""

    def __init__(self, source_name: str, frame_buffer: FrameBuffer):
        self._source_name = source_name
        self._buffer = frame_buffer
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ndi-recv")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        # Find the requested source
        finder = Finder()
        finder.open()

        source = None
        deadline = time.time() + 15.0
        while time.time() < deadline and self._running:
            finder.wait_for_sources(1.0)
            names = finder.get_source_names()
            for name in names:
                if self._source_name in name:
                    source = finder.get_source(name)
                    break
            if source is not None:
                break

        if source is None:
            logger.error("NDI source '%s' not found", self._source_name)
            finder.close()
            return

        logger.info("Connecting to NDI source: %s", source.name)

        # Create receiver
        receiver = Receiver(
            color_format=RecvColorFormat.BGRX_BGRA,
            bandwidth=RecvBandwidth.highest,
        )

        video_frame = VideoFrameSync()
        receiver.frame_sync.set_video_frame(video_frame)
        receiver.set_source(source)
        finder.close()

        logger.info("NDI receiver connected, capturing frames...")

        # Capture loop
        last_ts = -1.0
        while self._running:
            receiver.frame_sync.capture_video()
            if video_frame.xres == 0 or video_frame.yres == 0:
                time.sleep(0.005)
                continue

            ts = video_frame.get_timestamp_posix()
            if ts == last_ts:
                time.sleep(0.001)
                continue
            last_ts = ts

            frame_data = np.copy(
                video_frame.get_array().reshape(video_frame.yres, video_frame.xres, 4)
            )
            self._buffer.put(frame_data)

        receiver.disconnect()
        logger.info("NDI receiver stopped")

    @staticmethod
    def discover_sources(timeout_sec: float = 5.0) -> list:
        """Discover available NDI sources. Returns list of source name strings."""
        finder = Finder()
        finder.open()

        results = []
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            finder.wait_for_sources(1.0)
            results = finder.get_source_names()
            if results:
                break

        finder.close()
        return list(results)


# ---------------------------------------------------------------------------
# NDI Sender
# ---------------------------------------------------------------------------


class NDISender:
    """Sends processed frames out via NDI in a background thread."""

    def __init__(self, sender_name: str, output_queue: queue.Queue):
        self._sender_name = sender_name
        self._output_queue = output_queue
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ndi-send")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        sender = Sender(self._sender_name)
        send_frame = VideoSendFrame()
        send_frame.set_fourcc(FourCC.BGRX)
        send_frame.set_frame_rate(fractions.Fraction(TARGET_FPS, 1))
        sender.set_video_frame(send_frame)

        opened = False
        cur_w, cur_h = 0, 0
        buf = bytearray(0)
        mv = memoryview(buf)

        while self._running:
            try:
                img_bgrx = self._output_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            h, w = img_bgrx.shape[:2]
            if w == 0 or h == 0:
                continue

            # Re-allocate send buffer when resolution changes
            if w != cur_w or h != cur_h:
                cur_w, cur_h = w, h
                send_frame.set_resolution(w, h)
                if not opened:
                    sender.open()
                    opened = True
                    logger.info("NDI sender active as '%s'", self._sender_name)
                buf = bytearray(send_frame.get_data_size())
                mv = memoryview(buf)

            flat = img_bgrx.tobytes()
            buf[:len(flat)] = flat
            sender.write_video_async(mv)

        if opened:
            sender.close()
        logger.info("NDI sender stopped")


# ---------------------------------------------------------------------------
# NDIVideoTrack: aiortc VideoStreamTrack feeding Decart
# ---------------------------------------------------------------------------


class NDIVideoTrack(VideoStreamTrack):
    """Reads frames from FrameBuffer at the model's FPS and converts them
    for the Decart WebRTC connection.

    Handles:
      - Frame rate pacing (22 fps for mirage_v2)
      - Resolution scaling (source resolution -> 1280x704)
      - Color conversion (BGRX -> rgb24)
    """

    kind = "video"

    def __init__(self, frame_buffer: FrameBuffer):
        super().__init__()
        self._buffer = frame_buffer

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        """Pace frames at TARGET_FPS instead of the default 30 fps."""
        CLOCK_RATE = 90000
        PTIME = 1 / TARGET_FPS
        TIME_BASE = fractions.Fraction(1, CLOCK_RATE)

        if hasattr(self, "_ndi_ts"):
            self._ndi_ts += int(PTIME * CLOCK_RATE)
            wait = self._ndi_start + (self._ndi_ts / CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._ndi_start = time.time()
            self._ndi_ts = 0

        return self._ndi_ts, TIME_BASE

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        raw = self._buffer.get()

        if raw is None:
            # No frame yet — send a black frame
            img = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        else:
            # BGRX (H,W,4) -> BGR (H,W,3)
            img_bgr = raw[:, :, :3]

            # Resize if needed
            h, w = img_bgr.shape[:2]
            if w != TARGET_WIDTH or h != TARGET_HEIGHT:
                img_bgr = cv2.resize(
                    img_bgr, (TARGET_WIDTH, TARGET_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

            # BGR -> RGB
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


# ---------------------------------------------------------------------------
# RemoteStreamConsumer: Decart output -> NDI sender queue
# ---------------------------------------------------------------------------


class RemoteStreamConsumer:
    """Consumes frames from Decart's remote track, converts them for NDI,
    and pushes them into the output queue."""

    def __init__(self, output_queue: queue.Queue,
                 preview_buffer: Optional[FrameBuffer] = None):
        self._output_queue = output_queue
        self._preview_buffer = preview_buffer
        self._task: Optional[asyncio.Task] = None

    def on_remote_stream(self, track: MediaStreamTrack) -> None:
        """Callback passed to RealtimeConnectOptions.on_remote_stream."""
        logger.info("Remote stream received from Decart")
        self._task = asyncio.ensure_future(self._consume(track))

    async def _consume(self, track: MediaStreamTrack) -> None:
        while True:
            try:
                frame: VideoFrame = await track.recv()
            except MediaStreamError:
                logger.info("Remote stream ended")
                return

            # av.VideoFrame -> numpy RGB (H, W, 3)
            img_rgb = frame.to_ndarray(format="rgb24")

            # RGB -> BGRX for NDI
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_bgrx = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            # BGRA sets alpha=255 by default, which is what we want for BGRX

            # Push to output queue, dropping oldest if full
            try:
                self._output_queue.put_nowait(img_bgrx)
            except queue.Full:
                try:
                    self._output_queue.get_nowait()
                except queue.Empty:
                    pass
                self._output_queue.put_nowait(img_bgrx)

            # Also push to preview buffer for UI display
            if self._preview_buffer is not None:
                self._preview_buffer.put(img_bgrx)

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


async def select_ndi_source() -> str:
    """Discover NDI sources and let the user pick one interactively."""
    print("Discovering NDI sources...")
    sources = await asyncio.to_thread(NDIReceiver.discover_sources, 5.0)

    if not sources:
        print("No NDI sources found. Retrying (10s)...")
        sources = await asyncio.to_thread(NDIReceiver.discover_sources, 10.0)

    if not sources:
        print("Still no NDI sources found. Make sure an NDI source is running.")
        sys.exit(1)

    print("\nAvailable NDI sources:")
    for i, name in enumerate(sources, 1):
        print(f"  {i}. {name}")

    while True:
        choice = await asyncio.to_thread(input, "\nSelect source number: ")
        try:
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(sources):
                return sources[idx]
        except ValueError:
            pass
        print("Invalid selection, try again.")


async def get_prompt_from_user() -> str:
    """Get the initial style prompt interactively."""
    prompt = await asyncio.to_thread(
        input, "Enter style prompt (e.g. 'Cyberpunk city'): "
    )
    return prompt.strip() or "Anime style"


async def prompt_loop(realtime_client: RealtimeClient, shutdown_event: asyncio.Event) -> None:
    """Interactive loop: type a new prompt to change style, 'quit' to exit."""
    print("\n--- Type a new prompt to change style, or 'quit' to exit ---")
    while not shutdown_event.is_set():
        try:
            user_input = await asyncio.to_thread(input, "> ")
        except EOFError:
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        try:
            await realtime_client.set_prompt(user_input)
            print(f"  Prompt changed to: {user_input}")
        except Exception as e:
            print(f"  Error changing prompt: {e}")

    shutdown_event.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="NDI <-> Decart Mirage Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ndi_mirage_bridge.py --api-key sk-...\n"
            '  python ndi_mirage_bridge.py --source "MY-PC (OBS)" --prompt "Anime style"\n'
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DECART_API_KEY"),
        help="Decart API key (or set DECART_API_KEY env var)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="NDI source name (interactive selection if omitted)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Initial style prompt (interactive if omitted)",
    )
    parser.add_argument(
        "--sender-name",
        default="NDI-Mirage-Output",
        help="NDI output sender name (default: NDI-Mirage-Output)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.api_key:
        print("Error: Decart API key required. Use --api-key or set DECART_API_KEY.")
        sys.exit(1)

    # --- Source & prompt selection ---
    source_name = args.source or await select_ndi_source()
    prompt_text = args.prompt or await get_prompt_from_user()

    # --- Shared data structures ---
    frame_buffer = FrameBuffer()
    output_queue: queue.Queue = queue.Queue(maxsize=2)
    shutdown_event = asyncio.Event()

    # --- Start NDI receiver ---
    ndi_receiver = NDIReceiver(source_name, frame_buffer)
    ndi_receiver.start()

    print(f"Waiting for frames from '{source_name}'...")
    got_frame = await asyncio.to_thread(frame_buffer.wait_for_first_frame, 30.0)
    if not got_frame:
        print("Timed out waiting for NDI frames. Check source is active.")
        ndi_receiver.stop()
        return

    print("Receiving frames. Connecting to Decart...")

    # --- Start NDI sender ---
    ndi_sender = NDISender(args.sender_name, output_queue)
    ndi_sender.start()

    # --- Decart video track ---
    video_track = NDIVideoTrack(frame_buffer)

    # --- Remote stream consumer ---
    consumer = RemoteStreamConsumer(output_queue)

    # --- Connect to Decart Mirage ---
    model = models.realtime(DECART_MODEL)
    client = DecartClient(api_key=args.api_key)

    realtime_client = await RealtimeClient.connect(
        base_url=client.base_url,
        api_key=client.api_key,
        local_track=video_track,
        options=RealtimeConnectOptions(
            model=model,
            on_remote_stream=consumer.on_remote_stream,
            initial_state=ModelState(
                prompt=Prompt(text=prompt_text, enrich=True)
            ),
        ),
    )

    def on_connection_change(state):
        logger.info("Decart connection: %s", state)

    def on_error(error):
        logger.error("Decart error: %s", error)

    realtime_client.on("connection_change", on_connection_change)
    realtime_client.on("error", on_error)

    print()
    print("=" * 60)
    print("  NDI-Mirage Bridge is ACTIVE")
    print(f"  NDI Input:  {source_name}")
    print(f"  NDI Output: {args.sender_name}")
    print(f"  Model:      {DECART_MODEL} ({TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS}fps)")
    print(f"  Prompt:     {prompt_text}")
    print("=" * 60)

    # --- Interactive prompt loop (blocks until quit) ---
    try:
        await prompt_loop(realtime_client, shutdown_event)
    except (KeyboardInterrupt, EOFError):
        pass

    # --- Graceful shutdown ---
    print("\nShutting down...")
    await consumer.stop()
    await realtime_client.disconnect()
    ndi_sender.stop()
    ndi_receiver.stop()

    print("Done.")


if __name__ == "__main__":
    # aiortc requires SelectorEventLoop on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
