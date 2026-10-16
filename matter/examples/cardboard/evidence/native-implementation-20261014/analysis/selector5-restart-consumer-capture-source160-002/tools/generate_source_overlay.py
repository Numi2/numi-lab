#!/usr/bin/env python3
"""Build the source157-relative patch for the bounded consumer capture."""

from __future__ import annotations

import difflib
from pathlib import Path

ROOT = Path("/Users/home/cardboard-implementation-20261014")
BASE = ROOT / "snapshots/source-right-grip-telemetry-157"
PACKAGE = ROOT / "analysis/selector5-restart-consumer-capture-source160-001"
OUT = PACKAGE / "source-overlay.patch"

def diff_bytes(old: bytes, new: bytes, old_name: str, new_name: str) -> str:
    old_lines = old.decode("utf-8").splitlines(keepends=True)
    new_lines = new.decode("utf-8").splitlines(keepends=True)
    return "".join(difflib.unified_diff(
        old_lines, new_lines, fromfile=old_name, tofile=new_name, n=3))

runtime_base = (BASE / "matter/src/runtime.mm").read_bytes()
runtime_overlay = (PACKAGE / "source/matter/src/runtime.mm").read_bytes()
header_overlay = (PACKAGE / "source/matter/include/numi/matter/regional_consumer_capture.hpp").read_bytes()

patch = diff_bytes(runtime_base, runtime_overlay,
                   "a/matter/src/runtime.mm", "b/matter/src/runtime.mm")
header_lines = header_overlay.decode("utf-8").splitlines(keepends=True)
patch += "".join(difflib.unified_diff(
    [], header_lines,
    fromfile="/dev/null",
    tofile="b/matter/include/numi/matter/regional_consumer_capture.hpp",
    n=3))
OUT.write_text(patch)
print(OUT)
