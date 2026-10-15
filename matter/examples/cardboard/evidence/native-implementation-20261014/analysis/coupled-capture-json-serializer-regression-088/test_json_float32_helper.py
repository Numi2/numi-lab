#!/usr/bin/env python3
"""Compile and exercise the exact source-bound jsonFloat32Token helper."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

EXPECTED_RUNTIME_SHA256 = "a980ee59341429440e83984dd02073123c9647f19655a94260119a5d67aa5918"
DEFAULT_RUNTIME = Path("/Users/home/cardboard-implementation-20261014/analysis/coupled-capture-json-finite-serialization-088-overlay/runtime.mm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    raw = args.runtime.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_RUNTIME_SHA256:
        raise SystemExit(f"runtime source hash mismatch: {digest}")
    text = raw.decode()
    begin = text.index("[[nodiscard]] std::string jsonFloat32Token(")
    end = text.index("[[nodiscard]] std::string jsonQuoted(", begin)
    helper = text[begin:end].strip()
    if "std::isfinite(value)" not in helper or 'return "null";' not in helper:
        raise SystemExit("extracted helper no longer has the expected finite/null behavior")

    cpp = r'''#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
''' + helper + r'''
static void emit(const char* name, float value) {
    std::cout << "\"" << name << "\":{\"value\":" << jsonFloat32Token(value)
              << ",\"f32_bits_u32\":" << std::bit_cast<std::uint32_t>(value) << "}";
}
int main() {
    const float inf = std::numeric_limits<float>::infinity();
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    std::cout << "{";
    emit("max_finite", std::numeric_limits<float>::max()); std::cout << ",";
    emit("min_normal", std::numeric_limits<float>::min()); std::cout << ",";
    emit("denorm_min", std::numeric_limits<float>::denorm_min()); std::cout << ",";
    emit("positive_zero", 0.0f); std::cout << ",";
    emit("negative_zero", -0.0f); std::cout << ",";
    emit("positive_inf", inf); std::cout << ",";
    emit("negative_inf", -inf); std::cout << ",";
    emit("nan", qnan);
    std::cout << ",\"vascular_tolerance\":{\"value\":"
              << jsonFloat32Token(inf)
              << ",\"f32_bits_u32\":" << std::bit_cast<std::uint32_t>(inf)
              << ",\"semantics\":\"positive_infinity_not_applicable_no_vascular_unknowns\"}"
              << "}\n";
}
'''
    with tempfile.TemporaryDirectory(prefix="json-float32-helper-") as temp:
        cpp_path = Path(temp) / "helper_test.cpp"
        exe_path = Path(temp) / "helper_test"
        cpp_path.write_text(cpp)
        build = subprocess.run(
            ["clang++", "-std=c++23", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
             str(cpp_path), "-o", str(exe_path)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise SystemExit(f"compile failed:\n{build.stdout}")
        run = subprocess.run([str(exe_path)], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if run.returncode:
            raise SystemExit(f"helper executable failed: {run.stderr}")
        # Strict stdlib JSON parsing: no NaN/Infinity extensions accepted.
        doc = json.loads(run.stdout, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON constant {value}")))

    expect = {
        "max_finite": 0x7F7FFFFF,
        "min_normal": 0x00800000,
        "denorm_min": 0x00000001,
        "positive_zero": 0x00000000,
        "negative_zero": 0x80000000,
        "positive_inf": 0x7F800000,
        "negative_inf": 0xFF800000,
        "nan": 0x7FC00000,
        "vascular_tolerance": 0x7F800000,
    }
    for name, bits in expect.items():
        if doc[name]["f32_bits_u32"] != bits:
            raise SystemExit(f"raw FP32 word mismatch for {name}")
    for name in ("max_finite", "min_normal", "denorm_min", "positive_zero"):
        value = doc[name]["value"]
        if not isinstance(value, (int, float)):
            raise SystemExit(f"finite {name} did not serialize as a JSON number")
        restored = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        if restored != expect[name]:
            raise SystemExit(f"finite {name} JSON token did not round-trip to the same FP32 word")
    # JSON permits -0, but many parsers decode it as integer zero. Verify the
    # signed lexical token and retain the exact sign in the adjacent raw word.
    if not re.search(r'"negative_zero":\{\"value\":-0(?:\.0)?[,}]', run.stdout):
        raise SystemExit("negative zero did not retain its signed JSON token")
    if doc["negative_zero"]["value"] != 0:
        raise SystemExit("negative zero is not numerically zero")
    for name in ("positive_inf", "negative_inf", "nan", "vascular_tolerance"):
        if doc[name]["value"] is not None:
            raise SystemExit(f"nonfinite {name} must serialize as JSON null")
    if doc["vascular_tolerance"]["semantics"] != \
            "positive_infinity_not_applicable_no_vascular_unknowns":
        raise SystemExit("vascular tolerance semantic label mismatch")
    report = {
        "schema": "numi.matter.json_float32_helper_regression.v1",
        "runtime_sha256": digest,
        "helper_source": "extracted exact jsonFloat32Token function from runtime.mm",
        "strict_json_parse": True,
        "cases": {name: {"bits": bits, "value": doc[name]["value"]}
                  for name, bits in expect.items()},
        "negative_zero_token": "-0 is retained as a valid JSON number; exact sign is preserved in f32_bits_u32",
        "nonfinite_representation": "value=null with exact FP32 bits; vascular tolerance includes explicit not-applicable semantics",
        "pass": True,
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
