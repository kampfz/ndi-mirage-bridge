# NDI <-> Decart Mirage Bridge

## Architecture

```
NDI Source -> [NDI Receiver Thread] -> [FrameBuffer] -> [NDIVideoTrack (aiortc)]
    -> [Decart Mirage WebRTC] -> [on_remote_stream] -> [Frame Consumer Task]
    -> [OutputQueue] -> [NDI Sender Thread] -> NDI Output
```

## Components

1. **FrameBuffer** - Thread-safe latest-frame buffer (NDI receiver writes, VideoTrack reads)
2. **NDIReceiver** - Background thread capturing NDI frames via ndi-python
3. **NDISender** - Background thread sending processed frames back out via NDI
4. **NDIVideoTrack** - aiortc VideoStreamTrack subclass, bridges FrameBuffer to Decart (22fps, 1280x704, BGRX->RGB)
5. **RemoteStreamConsumer** - Async task consuming Decart output track, converts RGB->BGRX, pushes to OutputQueue
6. **CLI** - Source discovery, prompt input, interactive prompt changes
7. **main()** - Orchestrates everything with argparse

## Key Details

- mirage_v2: 22fps, 1280x704
- NDI uses BGRX (4ch), Decart/aiortc uses rgb24 (3ch)
- Latest-frame buffer pattern handles FPS mismatch (NDI 30/60fps -> Decart 22fps)
- Windows needs WindowsSelectorEventLoopPolicy for aiortc
- NDI frame data must be np.copy()'d before freeing
- ndi.initialize() called once in main, shared across threads

## Files

- `ndi_mirage_bridge.py` - Single-file application
- `requirements.txt` - Dependencies
