"""
NDI-Mirage Bridge — Tkinter GUI

Desktop GUI that wraps the core NDI/Decart pipeline from ndi_mirage_bridge.py.
Provides: NDI source picker, API key input, prompt presets, live preview,
and FPS/status monitoring.

Usage:
  python ndi_mirage_bridge_ui.py
"""

import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from aiortc import MediaStreamTrack
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

from decart import DecartClient, models
from decart.realtime import RealtimeClient, RealtimeConnectOptions
from decart.types import ModelState, Prompt

from ndi_mirage_bridge import (
    DECART_MODEL,
    TARGET_FPS,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    FrameBuffer,
    NDIReceiver,
    NDISender,
    NDIVideoTrack,
    RemoteStreamConsumer,
    SPOUT_AVAILABLE,
)

if SPOUT_AVAILABLE:
    from ndi_mirage_bridge import SpoutReceiver, SpoutSender

logger = logging.getLogger(__name__)

PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = 264
DEFAULT_OSC_PORT = 9000

STYLE_PRESETS = [
    "Cyberpunk",
    "Anime",
    "Watercolor",
    "Oil Painting",
    "Pixel Art",
    "Comic Book",
    "Pencil Sketch",
    "Neon Glow",
]

CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


class MirageBridgeApp(tk.Tk):
    """Main application window for the NDI-Mirage Bridge GUI."""

    def __init__(self):
        super().__init__()
        self.title("NDI-Mirage Bridge")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- State ---
        self._connected = False
        self._receiver: Optional[NDIReceiver] = None
        self._sender: Optional[NDISender] = None
        self._video_track: Optional[NDIVideoTrack] = None
        self._consumer: Optional[RemoteStreamConsumer] = None
        self._realtime_client: Optional[RealtimeClient] = None
        self._input_buffer: Optional[FrameBuffer] = None
        self._preview_buffer: Optional[FrameBuffer] = None
        self._output_queue: Optional[queue.Queue] = None
        self._connect_time: Optional[float] = None
        self._input_preview_photo: Optional[ImageTk.PhotoImage] = None
        self._output_preview_photo: Optional[ImageTk.PhotoImage] = None
        self._osc_transport = None

        # --- Asyncio event loop in background thread ---
        self._loop = asyncio.new_event_loop()
        if sys.platform == "win32":
            # aiortc requires SelectorEventLoop on Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            self._loop = asyncio.SelectorEventLoop()
        self._async_thread = threading.Thread(
            target=self._run_async_loop, daemon=True, name="asyncio"
        )
        self._async_thread.start()

        self._build_ui()
        self._update_preview()
        self._update_status()
        self.after(100, self._refresh_sources)

    # ------------------------------------------------------------------
    # Async thread
    # ------------------------------------------------------------------

    def _run_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit_async(self, coro):
        """Schedule a coroutine on the asyncio thread and return a Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # --- Top config frame ---
        config_frame = ttk.LabelFrame(self, text="Configuration", padding=8)
        config_frame.pack(fill="x", **pad)

        # API Key row
        row_api = ttk.Frame(config_frame)
        row_api.pack(fill="x", pady=2)
        ttk.Label(row_api, text="API Key:").pack(side="left")
        saved_key = os.getenv("DECART_API_KEY", "") or _load_config().get("api_key", "")
        self._api_key_var = tk.StringVar(value=saved_key)
        self._api_key_entry = ttk.Entry(row_api, textvariable=self._api_key_var, show="*", width=48)
        self._api_key_entry.pack(side="left", padx=(4, 2), fill="x", expand=True)
        self._show_key_var = tk.BooleanVar(value=False)
        self._show_key_btn = ttk.Checkbutton(
            row_api, text="Show", variable=self._show_key_var,
            command=self._toggle_key_visibility
        )
        self._show_key_btn.pack(side="left", padx=2)

        # Transport row
        row_transport = ttk.Frame(config_frame)
        row_transport.pack(fill="x", pady=2)
        ttk.Label(row_transport, text="Transport:").pack(side="left")
        transport_options = ["NDI"]
        if SPOUT_AVAILABLE:
            transport_options.append("Spout")
        self._transport_var = tk.StringVar(value=transport_options[0])
        self._transport_combo = ttk.Combobox(
            row_transport, textvariable=self._transport_var,
            state="readonly", values=transport_options, width=10,
        )
        self._transport_combo.pack(side="left", padx=(4, 2))
        self._transport_combo.bind("<<ComboboxSelected>>", self._on_transport_changed)

        # Source row
        row_ndi = ttk.Frame(config_frame)
        row_ndi.pack(fill="x", pady=2)
        self._source_label = ttk.Label(row_ndi, text="NDI Source:")
        self._source_label.pack(side="left")
        self._source_var = tk.StringVar()
        self._source_combo = ttk.Combobox(
            row_ndi, textvariable=self._source_var, state="readonly", width=44
        )
        self._source_combo.pack(side="left", padx=(4, 2), fill="x", expand=True)
        self._refresh_btn = ttk.Button(row_ndi, text="Refresh", command=self._refresh_sources)
        self._refresh_btn.pack(side="left", padx=2)

        # Output name row
        row_out = ttk.Frame(config_frame)
        row_out.pack(fill="x", pady=2)
        ttk.Label(row_out, text="Output Name:").pack(side="left")
        self._output_name_var = tk.StringVar(value="NDI-Mirage-Output")
        ttk.Entry(row_out, textvariable=self._output_name_var, width=48).pack(
            side="left", padx=(4, 0), fill="x", expand=True
        )

        # OSC Port row
        row_osc = ttk.Frame(config_frame)
        row_osc.pack(fill="x", pady=2)
        ttk.Label(row_osc, text="OSC Port:").pack(side="left")
        self._osc_port_var = tk.StringVar(value=str(DEFAULT_OSC_PORT))
        ttk.Entry(row_osc, textvariable=self._osc_port_var, width=8).pack(
            side="left", padx=(4, 4)
        )
        ttk.Label(row_osc, text="(0 = disabled, receives /mirage/prompt)",
                  foreground="gray").pack(side="left")

        # Auto Connect row
        row_auto = ttk.Frame(config_frame)
        row_auto.pack(fill="x", pady=2)
        self._auto_connect_var = tk.BooleanVar(
            value=_load_config().get("auto_connect", False)
        )
        ttk.Checkbutton(
            row_auto, text="Auto Connect", variable=self._auto_connect_var,
            command=self._save_auto_connect,
        ).pack(side="left")
        ttk.Label(
            row_auto, text="(auto-selects TouchDesigner source on launch)",
            foreground="gray",
        ).pack(side="left", padx=(4, 0))

        # --- Preview frame ---
        preview_frame = ttk.LabelFrame(self, text="Live Preview", padding=4)
        preview_frame.pack(fill="both", **pad)

        black = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), (0, 0, 0))

        # Input preview (left)
        self._input_frame = ttk.LabelFrame(preview_frame, text="NDI Input", padding=2)
        self._input_frame.pack(side="left", padx=(0, 4))
        self._input_preview_label = ttk.Label(self._input_frame, anchor="center")
        self._input_preview_label.pack()
        self._input_preview_photo = ImageTk.PhotoImage(black)
        self._input_preview_label.config(image=self._input_preview_photo)

        # Output preview (right)
        output_frame = ttk.LabelFrame(preview_frame, text="Mirage Output", padding=2)
        output_frame.pack(side="left", padx=(4, 0))
        self._output_preview_label = ttk.Label(output_frame, anchor="center")
        self._output_preview_label.pack()
        self._output_preview_photo = ImageTk.PhotoImage(black)
        self._output_preview_label.config(image=self._output_preview_photo)

        # --- Prompt frame ---
        prompt_frame = ttk.LabelFrame(self, text="Style Prompt", padding=8)
        prompt_frame.pack(fill="x", **pad)

        row_prompt = ttk.Frame(prompt_frame)
        row_prompt.pack(fill="x", pady=2)
        self._prompt_var = tk.StringVar(value="Cyberpunk city")
        ttk.Entry(row_prompt, textvariable=self._prompt_var, width=52).pack(
            side="left", padx=(0, 4), fill="x", expand=True
        )
        self._apply_btn = ttk.Button(row_prompt, text="Apply", command=self._apply_prompt)
        self._apply_btn.pack(side="left")

        # Preset buttons — two rows
        presets_frame1 = ttk.Frame(prompt_frame)
        presets_frame1.pack(fill="x", pady=(4, 0))
        presets_frame2 = ttk.Frame(prompt_frame)
        presets_frame2.pack(fill="x", pady=2)

        for i, preset in enumerate(STYLE_PRESETS):
            parent = presets_frame1 if i < 4 else presets_frame2
            btn = ttk.Button(
                parent, text=preset,
                command=lambda p=preset: self._on_preset_click(p)
            )
            btn.pack(side="left", padx=2)

        # --- Connect / Disconnect buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)
        self._connect_btn = ttk.Button(btn_frame, text="Connect", command=self._on_connect)
        self._connect_btn.pack(side="left", padx=(8, 4), expand=True, fill="x")
        self._disconnect_btn = ttk.Button(
            btn_frame, text="Disconnect", command=self._on_disconnect, state="disabled"
        )
        self._disconnect_btn.pack(side="left", padx=(4, 8), expand=True, fill="x")

        # --- Status bar ---
        status_frame = ttk.Frame(self, relief="sunken", padding=4)
        status_frame.pack(fill="x", side="bottom")
        self._status_var = tk.StringVar(value="Disconnected")
        self._input_fps_var = tk.StringVar(value="Input FPS: --")
        self._output_fps_var = tk.StringVar(value="Output FPS: --")
        self._timer_var = tk.StringVar(value="")

        ttk.Label(status_frame, textvariable=self._status_var).pack(side="left", padx=8)
        ttk.Separator(status_frame, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Label(status_frame, textvariable=self._input_fps_var).pack(side="left", padx=8)
        ttk.Separator(status_frame, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Label(status_frame, textvariable=self._output_fps_var).pack(side="left", padx=8)
        ttk.Separator(status_frame, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Label(status_frame, textvariable=self._timer_var).pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Key visibility
    # ------------------------------------------------------------------

    def _toggle_key_visibility(self):
        self._api_key_entry.config(show="" if self._show_key_var.get() else "*")

    def _save_auto_connect(self):
        cfg = _load_config()
        cfg["auto_connect"] = self._auto_connect_var.get()
        _save_config(cfg)

    # ------------------------------------------------------------------
    # Transport switching
    # ------------------------------------------------------------------

    def _on_transport_changed(self, event=None):
        transport = self._transport_var.get()
        self._source_label.config(text=f"{transport} Source:")
        self._input_frame.config(text=f"{transport} Input")
        self._source_combo.set("")
        self._source_combo["values"] = ()
        self._refresh_sources()

    # ------------------------------------------------------------------
    # Auto-connect on launch
    # ------------------------------------------------------------------

    def _try_auto_connect(self):
        """After discovery, auto-select a TouchDesigner source and connect."""
        if not self._auto_connect_var.get():
            return
        if not self._api_key_var.get().strip():
            return

        sources = list(self._source_combo["values"])
        td_source = None
        for s in sources:
            if "touchdesigner" in s.lower():
                td_source = s
                break

        if td_source is None:
            self._status_var.set("Auto-connect: no TouchDesigner source found")
            return

        self._source_var.set(td_source)
        self._on_connect()

    # ------------------------------------------------------------------
    # NDI source discovery
    # ------------------------------------------------------------------

    def _refresh_sources(self):
        transport = self._transport_var.get()
        self._refresh_btn.config(state="disabled")
        self._source_combo.set("")
        self._source_combo["values"] = ()
        self._status_var.set(f"Discovering {transport} sources...")

        def _discover():
            if transport == "Spout" and SPOUT_AVAILABLE:
                sources = SpoutReceiver.discover_sources()
            else:
                sources = NDIReceiver.discover_sources(timeout_sec=5.0)
            self.after(0, lambda: self._on_discovery_done(sources, transport))

        threading.Thread(target=_discover, daemon=True, name="discover").start()

    def _on_discovery_done(self, sources, transport):
        self._refresh_btn.config(state="normal")
        if sources:
            self._source_combo["values"] = sources
            self._source_combo.current(0)
            self._status_var.set(f"Found {len(sources)} {transport} source(s)")
            self._try_auto_connect()
        else:
            self._status_var.set(f"No {transport} sources found")

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------

    def _on_connect(self):
        api_key = self._api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("Error", "API Key is required.")
            return

        transport = self._transport_var.get()
        source = self._source_var.get().strip()
        if not source:
            messagebox.showerror("Error", f"Select a {transport} source first (click Refresh).")
            return

        output_name = self._output_name_var.get().strip() or "NDI-Mirage-Output"
        prompt_text = self._prompt_var.get().strip() or "Cyberpunk city"

        self._connect_btn.config(state="disabled")
        self._disconnect_btn.config(state="disabled")
        self._transport_combo.config(state="disabled")
        self._status_var.set("Connecting...")

        def _do_connect():
            try:
                # Shared data structures
                self._input_buffer = FrameBuffer()
                self._preview_buffer = FrameBuffer()
                self._output_queue = queue.Queue(maxsize=2)

                # Start receiver
                if transport == "Spout" and SPOUT_AVAILABLE:
                    self._receiver = SpoutReceiver(source, self._input_buffer)
                else:
                    self._receiver = NDIReceiver(source, self._input_buffer)
                self._receiver.start()

                # Wait for first frame
                self.after(0, lambda: self._status_var.set(f"Waiting for {transport} frames..."))
                got_frame = self._input_buffer.wait_for_first_frame(30.0)
                if not got_frame:
                    self._receiver.stop()
                    self.after(0, lambda: self._connect_failed(
                        f"Timed out waiting for {transport} frames. Check source is active."
                    ))
                    return

                # Start sender
                if transport == "Spout" and SPOUT_AVAILABLE:
                    self._sender = SpoutSender(output_name, self._output_queue)
                else:
                    self._sender = NDISender(output_name, self._output_queue)
                self._sender.start()

                # Create video track and consumer
                self._video_track = NDIVideoTrack(self._input_buffer)
                self._consumer = RemoteStreamConsumer(
                    self._output_queue, preview_buffer=self._preview_buffer,
                    input_buffer=self._input_buffer,
                )

                # Connect to Decart (async, non-blocking)
                self.after(0, lambda: self._status_var.set("Connecting to Decart..."))
                future = self._submit_async(self._async_connect(api_key, prompt_text))
                future.add_done_callback(self._on_decart_connect_done)

            except Exception as e:
                logger.exception("Connection setup failed")
                msg = str(e)
                self.after(0, lambda: self._connect_failed(msg))

        threading.Thread(target=_do_connect, daemon=True, name="connect").start()

    def _on_decart_connect_done(self, future):
        """Callback fired when the async Decart connection resolves."""
        try:
            future.result()
            self.after(0, self._connect_succeeded)
        except Exception as e:
            logger.error("Decart connection failed: %s", e)
            msg = str(e)
            self.after(0, lambda: self._connect_failed(msg))

    async def _async_connect(self, api_key: str, prompt_text: str):
        model = models.realtime(DECART_MODEL)
        client = DecartClient(api_key=api_key)

        self._realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=self._video_track,
            options=RealtimeConnectOptions(
                model=model,
                on_remote_stream=self._consumer.on_remote_stream,
                initial_state=ModelState(
                    prompt=Prompt(text=prompt_text, enrich=False)
                ),
            ),
        )

        def on_connection_change(state):
            logger.info("Decart connection: %s", state)

        def on_error(error):
            logger.error("Decart error: %s", error)

        self._realtime_client.on("connection_change", on_connection_change)
        self._realtime_client.on("error", on_error)

    def _connect_succeeded(self):
        self._connected = True
        self._connect_time = time.time()
        self._connect_btn.config(state="disabled")
        self._disconnect_btn.config(state="normal")
        self._status_var.set("Connected")
        # Start OSC server for remote prompt control
        self._start_osc_server()
        # Persist config for next launch
        cfg = _load_config()
        cfg["api_key"] = self._api_key_var.get().strip()
        _save_config(cfg)

    def _connect_failed(self, msg: str):
        self._connect_btn.config(state="normal")
        self._disconnect_btn.config(state="disabled")
        self._transport_combo.config(state="readonly")
        self._status_var.set("Disconnected")
        messagebox.showerror("Connection Failed", msg)

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    def _on_disconnect(self):
        if not self._connected:
            return
        self._disconnect_btn.config(state="disabled")
        self._status_var.set("Disconnecting...")

        def _do_disconnect():
            try:
                if self._consumer:
                    self._submit_async(self._consumer.stop()).result(timeout=10)
                if self._realtime_client:
                    self._submit_async(self._realtime_client.disconnect()).result(timeout=10)
                if self._sender:
                    self._sender.stop()
                if self._receiver:
                    self._receiver.stop()
            except Exception:
                logger.exception("Error during disconnect")
            finally:
                self.after(0, self._disconnect_done)

        threading.Thread(target=_do_disconnect, daemon=True, name="disconnect").start()

    def _disconnect_done(self):
        self._stop_osc_server()
        self._connected = False
        self._realtime_client = None
        self._consumer = None
        self._video_track = None
        self._receiver = None
        self._sender = None
        self._input_buffer = None
        self._preview_buffer = None
        self._output_queue = None
        self._connect_btn.config(state="normal")
        self._disconnect_btn.config(state="disabled")
        self._transport_combo.config(state="readonly")
        self._status_var.set("Disconnected")
        self._input_fps_var.set("Input FPS: --")
        self._output_fps_var.set("Output FPS: --")
        self._connect_time = None
        self._timer_var.set("")
        # Reset both previews to black
        black = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), (0, 0, 0))
        self._input_preview_photo = ImageTk.PhotoImage(black)
        self._input_preview_label.config(image=self._input_preview_photo)
        self._output_preview_photo = ImageTk.PhotoImage(black)
        self._output_preview_label.config(image=self._output_preview_photo)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _apply_prompt(self):
        if not self._connected or not self._realtime_client:
            return
        text = self._prompt_var.get().strip()
        if not text:
            return

        async def _set():
            try:
                await self._realtime_client.set_prompt(text)
                logger.info("Prompt changed to: %s", text)
            except Exception as e:
                logger.error("Error changing prompt: %s", e)

        self._submit_async(_set())

    def _on_preset_click(self, preset_text: str):
        self._prompt_var.set(preset_text)
        self._apply_prompt()

    # ------------------------------------------------------------------
    # OSC
    # ------------------------------------------------------------------

    def _handle_osc_prompt(self, address, *osc_args):
        """Called from the asyncio thread when an OSC /mirage/prompt message arrives."""
        if not osc_args:
            return
        new_prompt = str(osc_args[0])
        logger.info("OSC prompt received: %s", new_prompt)
        # Update the Tk StringVar on the main thread
        self.after(0, lambda: self._prompt_var.set(new_prompt))
        # Send to Decart
        if self._realtime_client:
            self._submit_async(self._realtime_client.set_prompt(new_prompt))

    def _start_osc_server(self):
        """Start the OSC server on the asyncio loop. Call after connection succeeds."""
        try:
            port = int(self._osc_port_var.get())
        except ValueError:
            port = 0
        if port <= 0:
            logger.info("OSC disabled (port=0)")
            return

        dispatcher = Dispatcher()
        dispatcher.map("/mirage/prompt", self._handle_osc_prompt)

        async def _serve():
            osc_server = AsyncIOOSCUDPServer(
                ("0.0.0.0", port), dispatcher, self._loop
            )
            transport, _ = await osc_server.create_serve_endpoint()
            return transport

        future = self._submit_async(_serve())
        try:
            self._osc_transport = future.result(timeout=5)
            logger.info("OSC server listening on 0.0.0.0:%d", port)
        except Exception as e:
            logger.error("Failed to start OSC server: %s", e)

    def _stop_osc_server(self):
        """Close the OSC transport if running."""
        if self._osc_transport is not None:
            self._osc_transport.close()
            self._osc_transport = None

    # ------------------------------------------------------------------
    # Preview update (~30 fps)
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox_preview(frame_bgrx: np.ndarray) -> Image.Image:
        """Resize a BGRX frame to fit the preview area, preserving aspect ratio."""
        img_rgb = cv2.cvtColor(frame_bgrx, cv2.COLOR_BGRA2RGB)
        h, w = img_rgb.shape[:2]
        scale = min(PREVIEW_WIDTH / w, PREVIEW_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
        x_off = (PREVIEW_WIDTH - new_w) // 2
        y_off = (PREVIEW_HEIGHT - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = img_rgb
        return Image.fromarray(canvas)

    def _update_preview(self):
        # Input preview (raw feed)
        if self._input_buffer is not None:
            frame = self._input_buffer.get()
            if frame is not None:
                try:
                    pil_img = self._letterbox_preview(frame)
                    self._input_preview_photo = ImageTk.PhotoImage(pil_img)
                    self._input_preview_label.config(image=self._input_preview_photo)
                except Exception:
                    pass

        # Output preview (Decart processed)
        if self._preview_buffer is not None:
            frame = self._preview_buffer.get()
            if frame is not None:
                try:
                    pil_img = self._letterbox_preview(frame)
                    self._output_preview_photo = ImageTk.PhotoImage(pil_img)
                    self._output_preview_label.config(image=self._output_preview_photo)
                except Exception:
                    pass

        self.after(33, self._update_preview)  # ~30 fps

    # ------------------------------------------------------------------
    # Status bar update (~2 Hz)
    # ------------------------------------------------------------------

    def _update_status(self):
        if self._connected:
            if self._input_buffer:
                self._input_fps_var.set(f"Input FPS: {self._input_buffer.fps:.1f}")
            if self._preview_buffer:
                self._output_fps_var.set(f"Output FPS: {self._preview_buffer.fps:.1f}")
            if self._connect_time is not None:
                elapsed = int(time.time() - self._connect_time)
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                self._timer_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        self.after(500, self._update_status)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def _on_close(self):
        if self._connected:
            self._on_disconnect()
            # Give disconnect a moment to finish
            self.after(2000, self._final_close)
        else:
            self._final_close()

    def _final_close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self.destroy()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = MirageBridgeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
