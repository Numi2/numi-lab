#!/usr/bin/env python3
"""Static source-order contract for the restart target reset repair.

This checks only dispatch ordering and buffer bindings. It is not a native
solver qualification or evidence that a restart mode is admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

BASE_SHA256 = "a7f1e72fd6b25fcc648ff4f68c3f03ce806988c712b696032058e6ba1d553d9f"
PATCHED_SHA256 = "836605dca3f6b264d5ee04a9f65b66997ecbdc39f60709039c7e1ac59becb087"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path: Path, expected: str | None) -> dict[str, object]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    actual = hashlib.sha256(raw).hexdigest()
    if expected and actual != expected:
        raise ValueError(f"source SHA-256 mismatch: expected {expected}, got {actual}")

    patterns = {
        "reset": r'dispatchGroups32\(\s*"nm_fgmres_regional_restart_reset_metadata"',
        "project_basis": r'dispatchThreads\(\s*"nm_fgmres_regional_project_basis"',
        "build_rap": r'dispatchGroups32\(\s*"nm_fgmres_regional_build_rap"',
        "capture_scale": r'dispatchGroups32\(\s*"nm_fgmres_regional_capture_restart_scale"',
    }
    positions = {}
    for name, pattern in patterns.items():
        matches = list(re.finditer(pattern, text))
        positions[name] = matches[0].start() if matches else -1
    if any(position < 0 for position in positions.values()):
        missing = [name for name, position in positions.items() if position < 0]
        raise ValueError(f"missing source dispatch markers: {missing}")
    if not (positions["reset"] < positions["project_basis"] <
            positions["build_rap"] < positions["capture_scale"]):
        raise ValueError(f"unsafe dispatch order: {positions}")

    if len(list(re.finditer(patterns["reset"], text))) != 1:
        raise ValueError("expected exactly one runtime reset dispatch")
    reset_end = text.find("state.femRegionalRestartGenerationInitialized.store(", positions["reset"])
    reset_barrier = text.find("[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];", positions["reset"])
    if not (positions["reset"] < reset_barrier < reset_end < positions["project_basis"]):
        raise ValueError("reset barrier/initialization marker does not precede basis projection")

    reset_slice = text[positions["reset"]:positions["project_basis"]]
    buffers = [
        "femRegionalRestartGeneration", "femRegionalRestartRawNormSquared",
        "femRegionalRestartModeScale", "femRegionalRestartModeValid",
        "femRegionalRestartTargetScale", "femRegionalAugmentedCoarseValid",
        "femRegionalRestartTailPreserved",
    ]
    if any(reset_slice.count(f"[encoder setBuffer:state.{name}\n") != 1
           for name in buffers):
        raise ValueError("reset dispatch buffer set changed or is incomplete")

    return {
        "source": str(path),
        "sha256": actual,
        "dispatch_order": ["reset_metadata", "project_basis", "build_rap", "capture_restart_scale"],
        "dispatch_offsets_bytes": positions,
        "reset_buffers": buffers,
        "reset_precedes_target_capture": True,
        "native_qualification": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    result = check(args.source, args.expected_sha256)
    import json
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
