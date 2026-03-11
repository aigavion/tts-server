#!/bin/bash
set -e
echo "=== Installing Kokoro TTS dependencies ==="
pip install kokoro>=0.9.4 soundfile websockets numpy
apt-get install -y espeak-ng
echo ""
echo "=== Setup complete ==="
echo "Run: python server_kokoro.py"
