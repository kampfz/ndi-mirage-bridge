# NDI-Mirage Bridge

Real-time AI style transfer for NDI video sources using [Decart Mirage](https://decart.ai/). Captures an NDI feed, processes it through Decart's generative model, and outputs the result as a new NDI source — all with a live side-by-side preview.

## Prerequisites

- **Python 3.11+**
- **An NDI video source** on your local network (e.g. OBS with NDI output, an NDI camera, NDI Tools screen capture)
- **A Decart API key** — get one at [decart.ai](https://decart.ai/)

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/kampfz/ndi-mirage-bridge.git
cd ndi-mirage-bridge
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the GUI

```bash
python ndi_mirage_bridge_ui.py
```

### 3. Connect

1. Paste your Decart API key (it will be saved for next time)
2. Click **Refresh** to discover NDI sources on your network
3. Select a source from the dropdown
4. Click **Connect**

The left preview shows your raw NDI input, the right shows the Decart Mirage output. The processed output is also sent as a new NDI source (default name: `NDI-Mirage-Output`) that you can pick up in any NDI-compatible application.

### 4. Change styles

Type a style prompt (e.g. "Watercolor painting") and click **Apply**, or use one of the preset buttons.

## Headless / CLI Mode

The core pipeline can also be run without the GUI:

```bash
python ndi_mirage_bridge.py
```

This will prompt you to select an NDI source interactively and connect via the terminal.

## Building a Standalone Executable

To distribute the app without requiring a Python installation:

```bash
pip install pyinstaller
pyinstaller ndi_mirage_bridge_ui.spec --noconfirm
```

The output will be in `dist/NDI-Mirage-Bridge/`. On macOS this produces a `.app` bundle.

> **Note (Windows):** Windows Defender may quarantine the built `.exe` as a false positive. Add `dist/` to your Defender exclusions before building.

## Architecture

```
NDI Source -> [NDI Receiver] -> [FrameBuffer] -> [NDIVideoTrack (aiortc)]
  -> [Decart Mirage via WebRTC] -> [RemoteStreamConsumer]
  -> [Output Queue] -> [NDI Sender] -> NDI Output
```

- **NDI Receiver** captures frames in a background thread
- **FrameBuffer** is a thread-safe single-slot buffer (latest frame wins, intermediate frames are dropped)
- **NDIVideoTrack** feeds frames to Decart at 22 fps via WebRTC
- **RemoteStreamConsumer** receives processed frames and routes them to NDI output and the UI preview
- **NDI Sender** broadcasts the output as a new NDI source

## Troubleshooting

- **No NDI sources found:** Make sure your NDI source is running on the same network and click Refresh again. NDI discovery can take a few seconds.
- **Connection timeout:** Decart's WebRTC connection may retry a few times — this is normal. If it fails repeatedly, check your API key and internet connection.
- **Low FPS:** The Decart Mirage model runs at ~22 fps. Input sources at higher frame rates are automatically downsampled.
