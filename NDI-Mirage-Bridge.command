#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python ndi_mirage_bridge_ui.py
