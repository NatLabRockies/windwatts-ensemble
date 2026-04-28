"""Timestamped logging helper used throughout the WEM pipeline."""

from __future__ import annotations

import sys
import time


def log(msg: str) -> None:
    """Print a timestamped message to stderr."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)
