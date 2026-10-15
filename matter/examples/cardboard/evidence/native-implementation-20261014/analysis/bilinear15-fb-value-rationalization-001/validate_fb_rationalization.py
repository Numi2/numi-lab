#!/usr/bin/env python3
"""Source-bound CPU checks for the opt-in bilinear-15 FB value rewrite.

This is a diagnostic replay, not native qualification. It binds compact copies
of the captured point records and the 294 per-iteration trace; upstream hashes
are recorded in selected-evidence.json.
"""
import csv
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile

import mpmath as mp

HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path("/Users/home/cardboard-implementation-20261014/snapshots/source-direct-projection-diagnostic-089")
EVIDENCE = HERE / "selected-evidence.json"
CSV_INPUT = HERE / "native294-trace-input.csv"
CPP = HERE / "fb_compensated_replay.cpp"
CPP_BIN = HERE / "fb_compensated_replay"
PATCHED = HERE / "constitutive_impl.cpp"
BASE_SOURCE = BASE / "matter/src/constitutive_impl.cpp"
HEADER = BASE / "matter/include/numi/matter/paper_compensated_bytecode.h"
INCLUDE = BASE / "include"
MATTER_INCLUDE = BASE / "matter/include"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bitwise_same_json(a, b):
    # JSON source numbers have already been serialized from the source-bound
    # FP32 records. Stable compact encoding detects any data change here.
    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
        b, sort_keys=True, separators=(",", ":")
    )


def load_jsonl_rows(records, policy):
    return {(r["environment"], r["tet_id"]): r
            for r in records["native_points"][policy]}


def f32_bits(v):
    import struct
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0]


def main():
    evidence = json.loads(EVIDENCE.read_text())
    for item in evidence["upstream"].values():
        if pathlib.Path(item["path"]).is_file():
            assert sha(item["path"]) == item["sha256"], item["path"]
    independent_comparison = json.loads(
        pathlib.Path(evidence["upstream"]["allpoint_fp64_comparison"]["path"]).read_text())
    independent_accurate = independent_comparison["policies"]["accurate"]
    assert independent_accurate["valid_rows"] == 3450
    assert independent_accurate["stress"]["rows_above_1e_minus_3"] == 0
    assert independent_accurate["tangent"]["rows_above_1e_minus_3"] == 26
    capture = {(r["environment"], r["tet_id"]): r
               for r in evidence["capture_tetrahedra"]}
    accurate = load_jsonl_rows(evidence, "accurate")
    default = load_jsonl_rows(evidence, "default")
    fail_ids = [(0, i) for i in (567, 568, 569, 1527, 1528, 1529)]
    control_ids = [(0, i) for i in (566, 570, 1526, 1530)]
    assert set(capture) == set(fail_ids + control_ids)
    assert set(accurate) == set(capture) and set(default) == set(capture)

    for key in fail_ids:
        a, d = accurate[key], default[key]
        assert a["stress_valid"] is False and a["tangent_valid"] is False
        assert a["tangent_projection_converged"] is False
        assert a["projection_failure"] == 7 and a["projection_iterations"] == 16
        assert d["stress_valid"] is True and d["tangent_valid"] is True
        assert d["tangent_projection_converged"] is True
    for key in control_ids:
        assert accurate[key]["stress_valid"] is True
        assert accurate[key]["tangent_valid"] is True
        assert accurate[key]["tangent_projection_converged"] is True
        assert default[key]["stress_valid"] is True

    # Check repeated element rows are truly the same captured local problem.
    def local_signature(row):
        c = row["center"]
        return (c["F_local"], c["F_local_low"], c["accepted_state"],
                c["Fdot_local_per_s"], row["material_parameter_count"],
                row["material_name"])

    fail_sig = {json.dumps(local_signature(capture[k]), sort_keys=True)
                for k in fail_ids}
    assert len(fail_sig) == 1
    control_a = local_signature(capture[(0, 566)])
    control_b = local_signature(capture[(0, 570)])
    assert control_a != control_b
    assert local_signature(capture[(0, 566)]) == local_signature(capture[(0, 1526)])
    assert local_signature(capture[(0, 570)]) == local_signature(capture[(0, 1530)])

    # Verify the native trace's exact high/low F words and accepted-state prefix
    # agree with the captured source object. The direct residual replay below
    # uses every native iteration state in the separately retained CSV.
    trace = evidence["trace_accurate294"]
    assert trace["capture_sha256"] == evidence["upstream"]["capture"]["sha256"]
    assert trace["point"]["environment"] == 0 and trace["point"]["tet_id"] == 1528
    point_words = trace["point"]["input_f32_bits"]
    cap = capture[(0, 1528)]["center"]
    expected_hi = [f32_bits(x) for row in cap["F_local"] for x in row]
    expected_lo = [f32_bits(x) for row in cap["F_local_low"] for x in row]
    assert point_words[:9] == expected_hi
    assert point_words[9:18] == expected_lo

    # Reproduce the exact two-word compensated evaluator used by production.
    # Its direct FB value must match all 16 recorded native per-iteration values.
    command = ["clang++", "-std=c++2b", f"-I{INCLUDE}", f"-I{MATTER_INCLUDE}",
               str(CPP), "-o", str(CPP_BIN)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    syntax = ["clang++", "-std=c++2b", "-fsyntax-only", f"-I{INCLUDE}",
              f"-I{MATTER_INCLUDE}", f"-I{BASE / 'matter/src'}", str(PATCHED)]
    subprocess.run(syntax, check=True, capture_output=True, text=True)
    with tempfile.TemporaryDirectory(prefix="fb-rationalization-apply-") as temp:
        target = pathlib.Path(temp) / "matter/src/constitutive_impl.cpp"
        target.parent.mkdir(parents=True)
        shutil.copyfile(BASE_SOURCE, target)
        subprocess.run(["patch", "-p1", "-i", str(HERE / "constitutive_impl.patch")],
                       cwd=temp, check=True, capture_output=True, text=True)
        assert sha(target) == sha(PATCHED)
    completed = subprocess.run([str(CPP_BIN), str(CSV_INPUT)], check=True,
                               capture_output=True, text=True)
    (HERE / "compensated-replay-output.csv").write_text(completed.stdout)
    replay_rows = list(csv.reader(completed.stdout.splitlines()))
    native_input_rows = list(csv.reader(CSV_INPUT.open()))
    assert len(replay_rows) == 16 and len(native_input_rows) == 16
    direct_errors = []
    selected_values = []
    replay_detail = []
    trace_values = trace["trace_f32"]
    trace_bits = trace["trace_f32_bits"]
    captured_parameters = evidence["capture_header"]["parameters_by_environment"][0][:28]
    for src, replay in zip(native_input_rows, replay_rows):
        assert int(float(src[0])) == int(replay[0])
        iteration = int(float(src[0]))
        trace_base = trace["trace_layout"]["header_slots"] + iteration * trace["trace_layout"]["iteration_slots"]
        state_high = list(map(float, src[30:45]))
        state_low = list(map(float, src[45:60]))
        for state in range(15):
            assert f32_bits(state_high[state]) == trace_bits[trace_base + 6 + state]
            assert f32_bits(state_low[state]) == trace_bits[trace_base + 22 + state]
        for j, parameter in enumerate(captured_parameters):
            assert f32_bits(float(src[2+j])) == f32_bits(parameter)
        native_value = float(src[1])
        traced_native_value = trace_values[trace_base + 38 + 12]
        assert f32_bits(native_value) == f32_bits(traced_native_value)
        direct_value = float(replay[7])
        selected_value = float(replay[11])
        positive_sum = float(replay[4])
        assert positive_sum >= 0.0
        direct_errors.append(abs(native_value - direct_value))
        selected_values.append(selected_value)
        replay_detail.append({
            "iteration": iteration,
            "a_compensated": float(replay[2]),
            "b_compensated": float(replay[3]),
            "a_plus_b_compensated": positive_sum,
            "native_direct_residual": native_value,
            "rationalized_selected_residual": selected_value,
            "difference": selected_value-native_value,
        })
    assert max(direct_errors) == 0.0
    assert all(math.isfinite(x) for x in selected_values)

    # Real-arithmetic identity and derivative check at high precision. Cases
    # cover the selected stable branch near cancellation, both sides of the
    # branch boundary, and a regular negative-sum control. This tests the
    # algebraic rewrite, not the generated GPU bytecode or local solve.
    mp.mp.dps = 90
    cases = [
        (mp.mpf("4.4e-16"), mp.mpf("0.3927852213"), mp.mpf("1e-3")),
        (mp.mpf("0"), mp.mpf("0.3927851915"), mp.mpf("1e-3")),
        (mp.mpf("1e-12"), mp.mpf("1e-3"), mp.mpf("1e-3")),
        (mp.mpf("-0.2"), mp.mpf("-0.3"), mp.mpf("1e-3")),
        (mp.mpf("1e-3"), mp.mpf("-1e-3"), mp.mpf("1e-3")),
    ]
    algebra_errors = []
    derivative_errors = []
    for a, b, eps in cases:
        radius = mp.sqrt(a*a + b*b + eps*eps)
        direct = radius - a - b
        stable = (eps*eps - 2*a*b) / (radius + a + b)
        algebra_errors.append(abs(direct - stable))
        fd_a = mp.diff(lambda x: mp.sqrt(x*x+b*b+eps*eps)-x-b, a)
        fd_b = mp.diff(lambda y: mp.sqrt(a*a+y*y+eps*eps)-a-y, b)
        stable_a = mp.diff(lambda x: (eps*eps-2*x*b) /
                           (mp.sqrt(x*x+b*b+eps*eps)+x+b), a)
        stable_b = mp.diff(lambda y: (eps*eps-2*a*y) /
                           (mp.sqrt(a*a+y*y+eps*eps)+a+y), b)
        derivative_errors.extend([abs(fd_a-stable_a), abs(fd_b-stable_b)])
    assert max(algebra_errors) < mp.mpf("1e-80")
    assert max(derivative_errors) < mp.mpf("1e-80")

    report = {
        "schema": "numi.cardboard.bilinear15-fb-rationalization-cpu-check.v1",
        "status": "pass_cpu_arithmetic_checks",
        "source_binding": {
            "base_snapshot": str(BASE),
            "base_constitutive_sha256": sha(BASE_SOURCE),
            "patched_constitutive_sha256": sha(PATCHED),
            "compensated_header_sha256": sha(HEADER),
            "selected_evidence_sha256": sha(EVIDENCE),
            "native_trace_csv_sha256": sha(CSV_INPUT),
            "upstream": evidence["upstream"],
        },
        "native296_subset": {
            "accurate_failure_count": len(fail_ids),
            "failure_ids": [f"{e}:{t}" for e, t in fail_ids],
            "default_failures_in_subset": 0,
            "accurate_control_ids": [f"{e}:{t}" for e, t in control_ids],
            "all_six_failures_same_captured_local_problem": True,
            "failure_point_input_equals_trace_capture_F_high_low": True,
        },
        "independent_native296_fp64_comparison": {
            "report_path": evidence["upstream"]["allpoint_fp64_comparison"]["path"],
            "report_sha256": evidence["upstream"]["allpoint_fp64_comparison"]["sha256"],
            "valid_accurate_rows": independent_accurate["valid_rows"],
            "stress_rows_over_1e_minus_3": independent_accurate["stress"]["rows_above_1e_minus_3"],
            "tangent_rows_over_1e_minus_3": independent_accurate["tangent"]["rows_above_1e_minus_3"],
            "worst_accurate_tangent_relative_l2": independent_accurate["tangent"]["worst_rows"][0]["metrics"]["tangent"]["relative_l2"],
            "zero_direction_rows": independent_accurate["zero_direction_rows"],
            "interpretation": "A separate one-direction pointwise issue remains and is not addressed by this liner FB value patch.",
        },
        "native294_replay": {
            "iterations": len(replay_rows),
            "csv_parameters_match_capture_header_f32": True,
            "csv_state_high_low_match_trace_f32_words": True,
            "csv_native_fb_row12_matches_trace_f32_word_each_iteration": True,
            "direct_compensated_value_matches_retained_trace_csv_exactly": True,
            "maximum_absolute_direct_value_difference": max(direct_errors),
            "positive_sum_branch_iterations": len(replay_detail),
            "maximum_absolute_rationalized_vs_direct_difference": max(
                abs(r["difference"]) for r in replay_detail),
            "per_iteration_values": replay_detail,
            "stable_branch_values_finite": True,
            "representative_native_terminal_status": trace["trace_status_code"],
            "patched_source_fsyntax_only_pass": True,
            "patch_applies_to_exact_base_sha": True,
        },
        "real_arithmetic_identity_test": {
            "precision_decimal_digits": mp.mp.dps,
            "cases": len(cases),
            "max_identity_abs_error_decimal": mp.nstr(max(algebra_errors), 12),
            "max_first_derivative_abs_error_decimal": mp.nstr(max(derivative_errors), 12),
            "eps_zero_policy": "not_transformed; accurate source admission requires positive epsilon",
        },
        "scope": {
            "what_this_establishes": "The source-ordered compensated FB value at the representative native trace suffers cancellation on the six captured failure inputs; the rationalized positive-sum formula is a high-precision identity with matching first derivatives.",
            "what_remains_unproven": [
                "patched native Metal results for all six points",
                "whether the local solver then satisfies the unchanged scaled-correction stop",
                "complete nonlinear solver behavior or performance",
                "material calibration or physical qualification",
                "exact failed run 288 candidate identity (capture283 is not that candidate)",
            ],
            "no_tolerance_or_constitutive_parameter_change": True,
            "authored_jacobian_and_stopping_policy_unchanged": True,
        },
    }
    (HERE / "cpu-validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"],
                      "failure_count": len(fail_ids),
                      "max_direct_replay_error": max(direct_errors),
                      "patched_sha256": sha(PATCHED)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise
