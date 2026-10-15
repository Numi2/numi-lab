#!/usr/bin/env python3
"""Read-only integrity/admission check for the sealed source-086 FEM capture."""
import collections
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

EXPECTED_CAPTURE = "4decb7be2fbfa93ad97c0f2f22cfaca974931269068bed90e98815b09066b870"
EXPECTED_LINER = "a2720b90f2004f2eccf180063e7ecc4e0acdda93f700c4fd357914fc24d2255a"
EXPECTED_MEDIUM = "76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9"
EXPECTED_SNAPSHOT = "769a5e34c1d919293cb263fd27fbd041a482d22016c2818eec41d44e47935752"
CAPTURE = Path("/Users/home/cardboard-implementation-20261014/solver/recycle-source-default-first-control-283/operator-capture")
SNAPSHOT = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-captured-point-and-contact-scope-086/frozen-source.json")
LINER = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-captured-point-and-contact-scope-086/matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter")
MEDIUM = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-captured-point-and-contact-scope-086/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter")

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def f32bits(x):
    return f"0x{struct.unpack('<I', struct.pack('<f', float(x)))[0]:08x}"

def flatten(x):
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(flatten(item))
        return out
    return [float(x)]

def main():
    capture_sha, snapshot_sha = sha(CAPTURE), sha(SNAPSHOT)
    liner_sha, medium_sha = sha(LINER), sha(MEDIUM)
    assert capture_sha == EXPECTED_CAPTURE, "capture SHA mismatch"
    assert snapshot_sha == EXPECTED_SNAPSHOT, "frozen-source manifest SHA mismatch"
    assert liner_sha == EXPECTED_LINER and medium_sha == EXPECTED_MEDIUM, "material asset SHA mismatch"
    rows = []
    counts = collections.Counter()
    param_rows = None
    header = None
    with CAPTURE.open() as stream:
        for line in stream:
            row = json.loads(line)
            if header is None:
                header = row
                assert header["schema"] == "numi.cardboard.fem-newton-capture.v1"
                assert header["root"] == 0 and header["newton_iteration"] == 0
                assert header["environment_count"] == 2 and header["tetrahedron_count_per_environment"] == 1920
                assert header["center_residual_source"] == "pre_fgmres_private_copy"
                assert "high_float32_plus_low_float32" in header["F_local_word_encoding"]
                for i, name in enumerate(("hajali2009_liner_hill_bilinear_fit", "hajali2009_medium_hill_bilinear_fit")):
                    assert header["materials"][i]["name"] == name
                    assert header["materials"][i]["index"] == i
                    assert header["materials"][i]["parameter_offset"] == 28 * i
                param_rows = header["parameters_by_environment"]
                assert len(param_rows) == 2 and len(param_rows[0]) == len(param_rows[1]) == 59
                assert [f32bits(x) for x in param_rows[0]] == [f32bits(x) for x in param_rows[1]]
                continue
            if row.get("record") != "tetrahedron" or row.get("material_index") not in (0, 1):
                continue
            mat = row["material_index"]
            expected_name = ("hajali2009_liner_hill_bilinear_fit", "hajali2009_medium_hill_bilinear_fit")[mat]
            assert row["material_name"] == expected_name and row["material_state_count"] == 15
            assert row["material_parameter_count"] == 28
            center = row["center"]
            for key, length in (("F_local", 9), ("F_local_low", 9), ("Fdot_local_per_s", 9),
                                ("F_direction_local_per_s", 9), ("rate_direction_local_per_s2", 9),
                                ("accepted_state", 15)):
                vals = flatten(center[key])
                assert len(vals) == length and all(math.isfinite(v) for v in vals)
            dtmeta = center["dt_seconds_bulk_modulus_kind_frame_active"]
            assert len(dtmeta) == 4 and dtmeta[0] > 0 and dtmeta[1] > 0
            counts[(row["environment"], mat)] += 1
            rows.append(row)
    assert len(rows) == 3456
    assert counts == {(0, 0): 1152, (0, 1): 576, (1, 0): 1152, (1, 1): 576}
    target = next(r for r in rows if r["environment"] == 0 and r["tet_id"] == 1528 and r["material_index"] == 0)
    center = target["center"]
    target_inputs = []
    for key in ("F_local", "F_local_low", "Fdot_local_per_s", "F_direction_local_per_s", "rate_direction_local_per_s2"):
        target_inputs.extend(flatten(center[key]))
    dt = center["dt_seconds_bulk_modulus_kind_frame_active"]
    target_inputs.extend((dt[0], dt[1]))
    shader = Path(__file__).with_name("cardboard_bilinear15_capture_diagnostic.metal").read_text()
    checker = Path(__file__).with_name("cardboard_bilinear15_capture_diagnostic_check.mm").read_text()
    common_patch = Path(__file__).with_name("common.metalinc.patch").read_text()
    trace_kernel = shader.split("kernel void cardboard_bilinear15_capture_trace", 1)[1]
    assert "output[41u+i]=projectionState[i]" in trace_kernel
    assert "output[56u+i]=projectionLow[i]" in trace_kernel
    assert "for(uint i=0u;i<15u;++i)" in trace_kernel
    assert "output[71u]=0.15625f" in trace_kernel and "output[72u]=0.3125f" in trace_kernel
    assert "output[summaryCount+i]=trace[i]" in trace_kernel and "summaryCount=73u" in trace_kernel
    assert "projectionIterationTrace[traceRecord+5u]=scaledCorrectionPolicy" in common_patch
    assert "? scaledCorrectionNorm : -1.0f;" in common_patch
    assert "--expected-capture-sha256" in checker and "capSHA==expectedCaptureSHA" in checker
    assert checker.index('runTrace(metal,defaults') < checker.index('if(nativeAll)') < checker.index('runBatch(metal,defaults')
    summary = {
        "schema": "numi.cardboard.bilinear15-capture-preflight.v1",
        "status": "pass",
        "capture_role": "retained first-control input from source-283; it is not asserted to be the exact failed source-288 candidate",
        "base_snapshot_frozen_source_sha256": snapshot_sha,
        "capture_sha256": capture_sha,
        "liner_asset_sha256": liner_sha,
        "medium_asset_sha256": medium_sha,
        "capture_rows": len(rows),
        "trace_state_low_high_packing_static_check": "pass",
        "scaled_correction_trace_slot_patch_static_check": "pass",
        "explicit_capture_sha_and_selected_first_static_check": "pass",
        "material_counts_by_environment": {f"{env}:{mat}": n for (env, mat), n in sorted(counts.items())},
        "selected_point": {"environment": 0, "tet_id": 1528, "material_index": 0,
                           "material_name": target["material_name"],
                           "input_float32_words": [f32bits(x) for x in target_inputs],
                           "accepted_state_float32_words": [f32bits(x) for x in center["accepted_state"]],
                           "dt_seconds": dt[0], "temperature_K": dt[1]},
        "physical_steps": 0,
        "native_execution": False,
        "qualification": False,
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("capture-preflight.json")
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "capture_rows": len(rows), "target": "env0/tet1528/liner",
                      "capture_sha256": capture_sha, "output": str(out)}, sort_keys=True))

if __name__ == "__main__":
    main()
