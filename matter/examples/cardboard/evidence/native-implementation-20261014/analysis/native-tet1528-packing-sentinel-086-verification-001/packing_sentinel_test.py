#!/usr/bin/env python3
"""Source-bound CPU sentinel check for the frozen 086 selected-point packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


EXPECTED_SHADER_SHA256 = "23f6f758cda7ccc0e0651e45a298644e6b71a3c25cd5ef2f7c9cbe4ceca40628"
EXPECTED_CHECKER_SHA256 = "27f1b7852668ccec459ee5d6060bf89da203358448aca0138c7e8ac53cdf9571"
EXPECTED_COMMON_SHA256 = "e53feb0402a461f355cf2ee5c3af89f95eb49176178e60d997d4a9cc6588a182"
STATE_CAPACITY = 32
ACTIVE_STATE_COUNT = 15
HIGH_BASE = 41
LOW_BASE = 56
SENTINEL_SLOTS = (71, 72)
TRACE_BASE = 73
TRACE_SLOTS = 5992
OUTPUT_SLOTS = TRACE_BASE + TRACE_SLOTS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_kernel(text: str, name: str) -> str:
    marker = f"kernel void {name}("
    start = text.find(marker)
    assert start >= 0, f"missing kernel {name}"
    brace = text.find("{", start)
    assert brace >= 0, f"missing body for {name}"
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i]
    raise AssertionError(f"unterminated kernel {name}")


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def float_from_bits(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def run(overlay: Path, base: Path) -> dict:
    shader_path = overlay / "cardboard_bilinear15_capture_diagnostic.metal"
    checker_path = overlay / "cardboard_bilinear15_capture_diagnostic_check.mm"
    common_path = overlay / "common.metalinc.source086.modified.txt"
    shared_path = base / "matter/include/numi/matter/shared.h"

    hashes = {
        "shader": sha256(shader_path),
        "checker": sha256(checker_path),
        "common_modified": sha256(common_path),
        "shared_header": sha256(shared_path),
    }
    assert hashes["shader"] == EXPECTED_SHADER_SHA256, "sealed shader hash changed"
    assert hashes["checker"] == EXPECTED_CHECKER_SHA256, "sealed checker hash changed"
    assert hashes["common_modified"] == EXPECTED_COMMON_SHA256, "sealed common source hash changed"

    shader = shader_path.read_text()
    checker = checker_path.read_text()
    common = common_path.read_text()
    header = shared_path.read_text()
    kernel = source_kernel(shader, "cardboard_bilinear15_capture_trace")

    assert re.search(r"#define\s+NM_MAX_MATERIAL_STATE\s+32u", header), "base capacity is not 32"
    assert ACTIVE_STATE_COUNT <= STATE_CAPACITY
    assert "for(uint i=0u;i<15u;++i){" in kernel
    assert "output[41u+i]=projectionState[i];" in kernel
    assert "output[56u+i]=projectionLow[i];" in kernel
    assert "output[71u]=0.15625f;" in kernel
    assert "output[72u]=0.3125f;" in kernel
    assert "constexpr uint summaryCount=73u;" in kernel
    assert "output[summaryCount+i]=trace[i];" in kernel
    assert "std::vector<float> out(73u+kTraceSlots,0.0f);" in checker
    assert "floatArrayJSON(f,out+73,kTraceSlots)" in checker
    assert "std::bit_cast<std::uint32_t>(data[71])==0x3e200000u" in checker
    assert "std::bit_cast<std::uint32_t>(data[72])==0x3ea00000u" in checker

    state_capacity = int(re.search(r"#define\s+NM_MAX_MATERIAL_STATE\s+(\d+)u", header).group(1))
    trace_state = int(re.search(r"kMaterialProjectionTraceStateCapacity\s*=\s*(\d+)u", common).group(1))
    trace_iters = int(re.search(r"kMaterialProjectionTraceIterationCapacity\s*=\s*(\d+)u", common).group(1))
    trace_record = int(re.search(r"kMaterialProjectionTraceRecordSlots\s*=\s*6u\s*\+\s*7u\s*\*\s*kMaterialProjectionTraceStateCapacity\s*\+\s*kMaterialProjectionTraceStateCapacity\s*\*\s*kMaterialProjectionTraceStateCapacity", common).group(0).count("+") and (6 + 7 * trace_state + trace_state * trace_state))
    trace_slots = int(re.search(r"kMaterialProjectionTraceSlots\s*=\s*kMaterialProjectionTraceHeaderSlots\s*\+\s*kMaterialProjectionTraceIterationCapacity\s*\*\s*kMaterialProjectionTraceRecordSlots", common).group(0).count("+") and (8 + trace_iters * trace_record))
    assert state_capacity == STATE_CAPACITY
    assert trace_state == 16 and trace_iters == 16 and trace_record == 374
    assert trace_slots == TRACE_SLOTS
    assert OUTPUT_SLOTS == 6065

    # Emulate destinations with a 32-state high/low source and a unique marker
    # in every output slot. Track writers so overlap is detected independently
    # of the output values themselves.
    untouched = 0xDEADBEEF
    words = [untouched] * OUTPUT_SLOTS
    writers: dict[int, str] = {}

    def write(index: int, value: int, writer: str) -> None:
        assert 0 <= index < OUTPUT_SLOTS, f"{writer} write outside output: {index}"
        assert index not in writers, f"overlapping write at slot {index}: {writers[index]} vs {writer}"
        writers[index] = writer
        words[index] = value

    # Reserve the diagnostic summary, sentinels and trace as the shader does.
    for i in range(41):
        write(i, 0x10000000 + i, "summary")
    write(71, f32_bits(0.15625), "sentinel-0")
    write(72, f32_bits(0.3125), "sentinel-1")

    high = [0x3F000000 + i for i in range(state_capacity)]
    low = [0x40000000 + i for i in range(state_capacity)]
    for i in range(ACTIVE_STATE_COUNT):
        write(HIGH_BASE + i, high[i], "high-state")
        write(LOW_BASE + i, low[i], "low-state")
    for i in range(TRACE_SLOTS):
        write(TRACE_BASE + i, 0x50000000 + i, "trace")

    assert words[71] == 0x3E200000 and words[72] == 0x3EA00000
    assert words[HIGH_BASE : HIGH_BASE + ACTIVE_STATE_COUNT] == high[:ACTIVE_STATE_COUNT]
    assert words[LOW_BASE : LOW_BASE + ACTIVE_STATE_COUNT] == low[:ACTIVE_STATE_COUNT]
    assert sum(writer == "high-state" for writer in writers.values()) == ACTIVE_STATE_COUNT
    assert sum(writer == "low-state" for writer in writers.values()) == ACTIVE_STATE_COUNT
    assert [i for i, writer in writers.items() if writer == "high-state"] == list(range(HIGH_BASE, HIGH_BASE + ACTIVE_STATE_COUNT))
    assert [i for i, writer in writers.items() if writer == "low-state"] == list(range(LOW_BASE, LOW_BASE + ACTIVE_STATE_COUNT))
    assert len(writers) == OUTPUT_SLOTS
    assert min(writers) == 0 and max(writers) == OUTPUT_SLOTS - 1

    return {
        "schema": "numi.cardboard.selected-trace-packing-sentinel.v1",
        "status": "pass",
        "classification": "static-source-and-CPU-packing-emulation; no native execution",
        "source_hashes": hashes,
        "layout": {
            "max_material_state": state_capacity,
            "active_projected_state_count": ACTIVE_STATE_COUNT,
            "high_state_slots": [HIGH_BASE, HIGH_BASE + ACTIVE_STATE_COUNT - 1],
            "low_state_slots": [LOW_BASE, LOW_BASE + ACTIVE_STATE_COUNT - 1],
            "sentinel_slots": list(SENTINEL_SLOTS),
            "trace_slots": [TRACE_BASE, TRACE_BASE + TRACE_SLOTS - 1],
            "trace_slot_count": trace_slots,
            "output_slot_count": OUTPUT_SLOTS,
            "backing_state_tail_not_serialized": list(range(ACTIVE_STATE_COUNT, state_capacity)),
        },
        "checks": [
            "sealed source hashes match",
            "active high and low state writers are bounded to 15 states",
            "state, sentinel, and trace destination sets are disjoint",
            "all 32-state backing capacity leaves states 15 through 31 outside this 15-state output",
            "host allocates, checks, and serializes the same 73+5992 slot trace range",
            "sentinel words remain 0x3e200000 and 0x3ea00000",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, default=Path("/Users/home/cardboard-implementation-20261014/analysis/native-tet1528-projection-diagnostic-086-overlay-001"))
    parser.add_argument("--base-snapshot", type=Path, default=Path("/Users/home/cardboard-implementation-20261014/snapshots/source-captured-point-and-contact-scope-086"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.overlay, args.base_snapshot)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
