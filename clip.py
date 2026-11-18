#!/usr/bin/env python3
"""
Mini OpusClip: long YouTube video -> N Instagram-ready clips (4:5, captions, < 60s).

Pipeline: yt-dlp -> faster-whisper (word timestamps) -> Gemini (picks hook moments)
          -> OpenCV face-centred 4:5 crop -> ffmpeg burns captions.

Usage:
    export GEMINI_API_KEY=...
    python clip.py "https://youtu.be/XXXX" --n 5
    python clip.py --file myvideo.mp4 --n 5
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

OUT_W, OUT_H = 1080, 1350  # Instagram 4:5
MAX_CLIP_SEC = 58.0  # keep safely under 60
MIN_CLIP_SEC = 15.0


# ----------------------------------------------------------------------------
# 1. Download
# ----------------------------------------------------------------------------
def download(url: str, work: Path) -> Path:
    out = work / "source.mp4"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
