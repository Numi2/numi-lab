#!/usr/bin/env python3
"""Compile and execute the production barrier JSON writer, then parse both modes."""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile


def main() -> int:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    runtime = (source_root / "matter/src/runtime.mm").read_text()
    call = """detail::appendDeformableLimitTraceBarrier(
                                line, record.barrier.x, record.barrier.y,
                                record.barrier.z, record.barrier.w,
                                capturedAppendFloat);"""
    assert runtime.count(call) == 1, "production trace does not call tested barrier writer"
    assert 'line << ",\\\"barrier\\\":[";' not in runtime, \
        "inline production barrier writer reappeared"

    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("C++ compiler unavailable for serializer roundtrip")

    harness = r"""
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include "numi/matter/deformable_limit_trace_json.hpp"

static void appendFloat(std::ostringstream& out, float value) {
    out << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
}
static std::string emit(bool attachPerAlpha) {
    std::ostringstream line;
    line << "{\"schema\":1,\"barrier_owner\":\"record\"";
    numi::matter::detail::appendDeformableLimitTraceBarrier(
        line, -5.00007809e-6f, 3.02097760e-6f, 2.49543817e-8f,
        4.99992166e-6f, appendFloat);
    if (attachPerAlpha)
        line << ",\"per_alpha_trace\":{\"schema\":1,\"trials\":[{\"index\":0}]}";
    line << "}";
    return line.str();
}
int main() {
    std::cout << emit(false) << '\n' << emit(true) << '\n';
}
"""
    with tempfile.TemporaryDirectory(prefix="deformable-trace-json-") as td:
        root = pathlib.Path(td)
        cpp = root / "roundtrip.cpp"
        exe = root / "roundtrip"
        cpp.write_text(harness)
        subprocess.run(
            [compiler, "-std=c++17", "-I", str(source_root / "matter/include"),
             str(cpp), "-o", str(exe)],
            check=True, capture_output=True, text=True)
        output = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    rows = [json.loads(line) for line in output.splitlines()]
    assert len(rows) == 2
    assert rows[0]["barrier"] == [
        -5.00007809e-6, 3.02097760e-6, 2.49543817e-8, 4.99992166e-6
    ]
    assert "per_alpha_trace" not in rows[0]
    assert rows[1]["barrier"] == rows[0]["barrier"]
    assert rows[1]["per_alpha_trace"]["trials"] == [{"index": 0}]
    print("PASS: production barrier writer round-trips with per-alpha absent and present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
