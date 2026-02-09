# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

_project = Path(SPECPATH)  # SPECPATH is set by PyInstaller to the spec file's dir

# Locate site-packages for the current platform's venv layout
if sys.platform == "win32":
    site_packages = _project / ".venv" / "Lib" / "site-packages"
else:
    # macOS / Linux: .venv/lib/pythonX.Y/site-packages
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = _project / ".venv" / "lib" / py_ver / "site-packages"

# SpoutGL native libraries (Windows-only)
spout_binaries = []
if sys.platform == "win32":
    spout_dir = site_packages / "SpoutGL"
    if spout_dir.exists():
        spout_binaries += [(str(f), "SpoutGL") for f in spout_dir.glob("*.pyd")]
        spout_binaries += [(str(f), "SpoutGL") for f in spout_dir.glob("*.dll")]

# cyndilib native libraries
cyndilib_bin = site_packages / "cyndilib" / "wrapper" / "bin"
cyndilib_dest = os.path.join("cyndilib", "wrapper", "bin")

cyndilib_wrapper_dest = os.path.join("cyndilib", "wrapper")

if sys.platform == "win32":
    ndi_libs = [(str(f), cyndilib_dest) for f in cyndilib_bin.glob("*.dll")]
    cyndilib_exts = [(str(f), "cyndilib")
                     for f in (site_packages / "cyndilib").glob("*.pyd")]
    cyndilib_exts += [(str(f), cyndilib_wrapper_dest)
                      for f in (site_packages / "cyndilib" / "wrapper").glob("*.pyd")]
elif sys.platform == "darwin":
    ndi_libs = [(str(f), cyndilib_dest) for f in cyndilib_bin.glob("*.dylib")]
    cyndilib_exts = [(str(f), "cyndilib")
                     for f in (site_packages / "cyndilib").glob("*.so")]
    cyndilib_exts += [(str(f), cyndilib_wrapper_dest)
                      for f in (site_packages / "cyndilib" / "wrapper").glob("*.so")]
else:
    ndi_libs = [(str(f), cyndilib_dest) for f in cyndilib_bin.glob("*.so*")]
    cyndilib_exts = [(str(f), "cyndilib")
                     for f in (site_packages / "cyndilib").glob("*.so")]
    cyndilib_exts += [(str(f), cyndilib_wrapper_dest)
                      for f in (site_packages / "cyndilib" / "wrapper").glob("*.so")]

a = Analysis(
    ["ndi_mirage_bridge_ui.py"],
    pathex=[],
    binaries=ndi_libs + cyndilib_exts + spout_binaries,
    datas=[],
    hiddenimports=[
        "cyndilib",
        "cyndilib.wrapper",
        "cyndilib.finder",
        "cyndilib.receiver",
        "cyndilib.sender",
        "cyndilib.video_frame",
        "cyndilib.audio_frame",
        "cyndilib.framesync",
        "cyndilib.locks",
        "cyndilib.buffertypes",
        "cyndilib.callback",
        "cyndilib.metadata_frame",
        "cyndilib.audio_reference",
        "cyndilib.send_frame_status",
        "cyndilib.wrapper.common",
        "cyndilib.wrapper.ndi_recv",
        "cyndilib.wrapper.ndi_send",
        "cyndilib.wrapper.ndi_structs",
        "decart",
        "decart.realtime",
        "aiortc",
        "pythonosc",
        "pythonosc.dispatcher",
        "pythonosc.osc_server",
        "SpoutGL",
        "SpoutGL.enums",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NDI-Mirage-Bridge",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # No console window (Windows: no cmd, macOS: .app-style)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="NDI-Mirage-Bridge",
)

# macOS: wrap in a .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="NDI-Mirage-Bridge.app",
        bundle_identifier="com.ndi-mirage-bridge.app",
    )
