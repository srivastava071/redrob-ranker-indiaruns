"""
Small file-reading helpers.

The candidate file can be very large, so we read it line by line instead of
loading the whole file into memory at once.
"""

from __future__ import annotations

import gzip
import io
import json
from typing import Iterator, Dict, Any


def open_maybe_gzip(path: str) -> io.TextIOBase:
    """Open normal .jsonl files and compressed .jsonl.gz files the same way."""
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one candidate dictionary at a time."""
    with open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
