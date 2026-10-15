#!/usr/bin/env python3
"""Small synthetic parser/failure-injection tests; not native kernel evidence."""
from __future__ import annotations

import struct
import unittest

from verify_refinement_output import verify


def packed(fmt: str, values) -> bytes:
    return struct.pack("<" + fmt * len(values), *values)


def synthetic_bundles():
    # One one-node patch, 3x3 identity A, zero coarse basis, and b=(1,0,0).
    h = (1, 1, 1, 1, 1, 1, 3, 9, 3, 2, 0)
    inp = bytearray(b"NMSGKC1\0" + struct.pack("<11I", *h))
    inp += packed("f", (1.0,))                         # RAP LU
    inp += packed("I", (0,))                           # RAP pivots
    inp += packed("I", (1,))                           # RAP valid
    inp += packed("f", (1.0, 0.0, 0.0, 0.0))          # basis
    inp += packed("f", (0.0, 0.0, 0.0, 0.0))          # regional basis
    inp += packed("f", (0.0, 0.0, 0.0, 0.0))          # regional AP
    inp += packed("I", (0, 1, 0, 0))                   # patch descriptor
    inp += packed("I", (0,))                           # patch node
    inp += packed("I", (0, 1))                         # node patch offsets
    inp += packed("I", (0,))                           # node patch value
    inp += packed("I", (1,))                           # node coverage
    inp += packed("f", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    out = bytearray(b"NMSRC001" + struct.pack("<8I", 1, 1, 1, 1, 1, 9, 3, 10))
    out += packed("f", (0.0,))                         # coarse RHS
    out += packed("f", (0.0,))                         # coefficient
    out += packed("I", (1,))                           # factor valid
    out += packed("f", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    out += packed("I", (0, 1, 2))                      # LU pivots
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # mode0
    out += packed("f", (1.0, 0.0, 0.0, 0.0))          # mode1
    out += packed("f", (1.0, 0.0, 0.0, 0.0))          # preconditioned0
    out += packed("f", (1.0, 0.0, 0.0, 0.0))          # preconditioned1
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # zero mode0 correction
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # zero mode0 preconditioned
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # zero correction
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # zero preconditioned
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # completed correction
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # completed preconditioned
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # invalid factor correction
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # invalid mode correction
    out += packed("I", (0,))                           # singular valid
    out += packed("f", (0.0,) * 9)                     # singular LU
    out += packed("I", (0, 1, 2))                      # singular pivots
    out += packed("f", (0.0, 0.0, 0.0, 0.0))          # singular correction
    status_codes = (0, 0, 0, 0, 0, 0, 7, 1, 7, 7)
    for code in status_codes:
        out += packed("I", (code, 0, 0, 0, 0, 0, 0, 0))
        out += packed("f", (0.0, 0.0, 0.0, 0.0))
    return bytes(inp), bytes(out)


class RefinementVerifierTests(unittest.TestCase):
    def test_synthetic_contract_fixture_passes(self):
        input_raw, output_raw = synthetic_bundles()
        report = verify(input_raw, output_raw)
        self.assertTrue(report["pass"], report)

    def test_truncated_output_fails_closed(self):
        input_raw, output_raw = synthetic_bundles()
        with self.assertRaisesRegex(ValueError, "truncated"):
            verify(input_raw, output_raw[:-1])

    def test_wrong_refined_solution_fails_residual_gate(self):
        input_raw, output_raw = synthetic_bundles()
        # Output layout: header + coarse(4) + coeff(4) + factor-valid(4) + LU(36)
        # + pivots(12) + mode0(16), then mode1 begins.
        mode1_first = 8 + 32 + 4 + 4 + 4 + 36 + 12 + 16
        bad = bytearray(output_raw)
        struct.pack_into("<f", bad, mode1_first, 0.0)
        report = verify(input_raw, bytes(bad))
        self.assertFalse(report["pass"])
        gate = next(item for item in report["checks"]
                    if item["name"] == "refined_all_patch_residual_gate")
        self.assertFalse(gate["pass"])

    def test_unexpected_status_fails(self):
        input_raw, output_raw = synthetic_bundles()
        # Factor status is the first of nine 48-byte status records at output end.
        bad = bytearray(output_raw)
        status_offset = len(bad) - 10 * 48
        struct.pack_into("<I", bad, status_offset, 7)
        report = verify(input_raw, bytes(bad))
        self.assertFalse(report["pass"])


if __name__ == "__main__":
    unittest.main()
