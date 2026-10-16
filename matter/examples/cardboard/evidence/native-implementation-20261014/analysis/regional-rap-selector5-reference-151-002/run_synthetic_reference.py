#!/usr/bin/env python3
"""Run bounded synthetic algebraic checks and emit their numeric summary."""
from __future__ import annotations
import json
import math
from pathlib import Path

import regional_rap_reference as rap
import selector5_graph_reference as graph
from test_selector5_graph_reference import fixture


def main() -> None:
    capture = fixture()
    x = [0.2, -0.4, 0.8, 0.1, -0.3, 0.5, 7.0, 8.0, -9.0]
    y = [-0.7, 0.9, 0.3, 0.6, -0.2, 0.1, 4.0, -5.0, 3.0]
    sx = graph.apply_base_S(capture, x)
    sty = graph.apply_base_S_transpose(capture, y)
    hx = graph.apply_H(capture, x)
    hty = graph.apply_H_transpose(capture, y)
    report = {
        "schema": "numi.matter.selector5_graph_reference_synthetic.v1",
        "scope": "synthetic binary64 algebraic reference; no native execution",
        "overlapping_patches": len(capture.patches),
        "patch_membership_count": len(capture.patch_nodes),
        "shared_node_coverage": capture.node_coverage[1],
        "base_mode_count": capture.mode_count,
        "H_adjoint_dot_defect": abs(rap.dot(hx, y) - rap.dot(x, hty)),
        "S_adjoint_dot_defect": abs(rap.dot(sx, y) - rap.dot(x, sty)),
        "base_RAP_factor": graph.compare_base_rap_to_ptap(capture, atol=1e-12),
        "patch_factor_reconstruction": [
            graph.factor_reconstruction_error(p.original, p.factor)
            for p in capture.patches
        ],
        "fixed_output_is_zero": sx[6:] == [0.0, 0.0, 0.0]
            and sty[6:] == [0.0, 0.0, 0.0],
        "augmented_selected_view_supported": False,
        "native_selector5_apply_verified": False,
    }
    out = Path(__file__).resolve().parent / "synthetic-reference.json"
    out.write_text(json.dumps(report, sort_keys=True, indent=2,
                              allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
