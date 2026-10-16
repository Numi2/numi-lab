#!/usr/bin/env python3
import hashlib, json, pathlib
import numpy as np
Q = pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
A = Q / "analysis"
OUT = A / "native547-critical23-sweep-input-audit-008"
if (OUT / "manifest.json").exists(): raise SystemExit("sealed audit package; refusing overwrite")
CENTER_PATH = A / "native541-center-input-001/input.json"
CENTER = json.loads(CENTER_PATH.read_text())
CENTER_ROWS = {(int(r["environment"]), int(r["tet_id"])): r for r in CENTER["rows"]}
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def f32(values): return np.asarray(values, dtype=np.float32)
def words(a): return np.asarray(a, dtype=np.float32).view(np.uint32)
def audit(label, folder, expected_rows):
    folder = A / folder
    input_path, ref_path = folder / "input.json", folder / "stable-fp64-reference.jsonl"
    raw = json.loads(input_path.read_text())
    refs = [json.loads(line) for line in ref_path.read_text().splitlines()]
    if len(raw["rows"]) != expected_rows or len(refs) != expected_rows: raise AssertionError(f"{label}: row count")
    if raw["source_capture_sha256"] != CENTER["source_capture_sha256"]: raise AssertionError(f"{label}: capture binding")
    if raw["compiled_package_sha256"] != CENTER["compiled_package_sha256"]: raise AssertionError(f"{label}: package binding")
    if not np.array_equal(words(raw["parameters_by_material"]), words(CENTER["parameters_by_material"])): raise AssertionError(f"{label}: parameter words")
    endpoints=[]; centers=0; accepted_changes=0; ref_center_mismatch=0; max_ref_center_mismatch=0.0; direction_errors=[]; rate_errors=[]; off_axis_errors=[]; stored_realized_errors=[]
    for i,(row,ref) in enumerate(zip(raw["rows"],refs)):
        if int(ref["index"]) != i: raise AssertionError(f"{label}: index mismatch {i}")
        env,tet=int(row["environment"]),int(row["tet_id"]); source=CENTER_ROWS[(env,tet)]
        w,sw=f32(row["packed_input_f32"]),f32(source["packed_input_f32"])
        if not np.array_equal(words(f32(row["accepted_state_f32"])),words(f32(source["accepted_state_f32"]))): accepted_changes+=1
        center_F=(sw[:9].astype(np.float64)+sw[9:18].astype(np.float64)).reshape(3,3)
        actual_F=(w[:9].astype(np.float64)+w[9:18].astype(np.float64)).reshape(3,3)
        referr=float(np.max(np.abs(np.asarray(ref["center_F"],dtype=np.float64)-center_F)))
        if referr != 0.0: ref_center_mismatch += 1
        max_ref_center_mismatch=max(max_ref_center_mismatch,referr)
        if ref["label"]!="endpoint":
            centers+=1
            if not np.array_equal(actual_F,center_F): raise AssertionError(f"{label}: center/repeat F changed row {i}")
            continue
        col,sign=int(ref["basis_column"]),int(ref["sign"]); eps=float(ref["epsilon"])
        if not 0<=col<9 or sign not in (-1,1) or eps<=0: raise AssertionError(f"{label}: malformed endpoint {i}")
        H=np.zeros((3,3),dtype=np.float32); H.flat[col]=1.0; dt=float(sw[45])
        direction_errors += [float(np.max(np.abs(w[27:36]-H.reshape(9)))),float(np.max(np.abs(w[36:45]-(H/dt).reshape(9))))]
        target_F=center_F.copy(); target_F.flat[col]+=sign*eps
        delta_F=actual_F-center_F; delta=float(delta_F.flat[col]); off=delta_F.copy(); off.flat[col]=0.0
        off_axis_errors.append(float(np.max(np.abs(off))))
        stored_realized_errors.append(float(np.max(np.abs(actual_F-np.asarray(ref["realized_F"],dtype=np.float64)))))
        base_rate=sw[18:27].astype(np.float64); actual_rate=w[18:27].astype(np.float64)
        ideal_rate=base_rate.copy(); ideal_rate[col]+=sign*eps/dt
        rate_errors.append(float(np.max(np.abs(actual_rate-ideal_rate))))
        endpoints.append({"index":i,"tet":tet,"basis_column":col,"power":int(ref["power"]),"sign":sign,
          "epsilon":eps,"realized_component_delta_F":delta,"target_component_delta_F":sign*eps,
          "target_abs_error":float(abs(actual_F.flat[col]-target_F.flat[col])),
          "full_F_max_error":float(np.max(np.abs(actual_F-target_F))),"zero_realized":delta==0.0,
          "wrong_or_zero_sign":delta*sign<=0.0,"rate_component_delta":float(actual_rate.flat[col]-base_rate[col]),
          "ideal_rate_delta":sign*eps/dt})
    powers={}
    for e in endpoints:
        d=powers.setdefault(str(e["power"]),{"count":0,"zero":0,"wrong_or_zero_sign":0,"max_abs_target_error":0.0})
        d["count"]+=1; d["zero"]+=int(e["zero_realized"]); d["wrong_or_zero_sign"]+=int(e["wrong_or_zero_sign"])
        d["max_abs_target_error"]=max(d["max_abs_target_error"],e["target_abs_error"])
    return {"label":label,"folder":str(folder),"input_sha256":sha(input_path),"reference_sha256":sha(ref_path),"builder_sha256":sha(folder/"build_input.py") if (folder/"build_input.py").exists() else None,
      "rows":len(raw["rows"]),"centers_and_repeats":centers,"endpoint_rows":len(endpoints),
      "accepted_state_word_change_rows":accepted_changes,"reference_center_mismatch_rows":ref_center_mismatch,"max_reference_center_mismatch":max_ref_center_mismatch,"zero_endpoint_dF":sum(e["zero_realized"] for e in endpoints),
      "wrong_or_zero_sign_endpoint_dF":sum(e["wrong_or_zero_sign"] for e in endpoints),
      "max_full_F_target_error":max((e["full_F_max_error"] for e in endpoints),default=0.0),
      "max_off_axis_dF":max(off_axis_errors,default=0.0),
      "max_reference_realized_F_mismatch":max(stored_realized_errors,default=0.0),
      "max_direction_word_value_error":max(direction_errors,default=0.0),
      "max_rate_delta_vs_ideal":max(rate_errors,default=0.0),
      "example_tet740_col0_p10_minus":[e for e in endpoints if (e["tet"],e["basis_column"],e["power"],e["sign"])==(740,0,10,-1)],
      "per_power":powers}
results=[audit("superseded_547_v1","native547-critical23-sweep-input-001",1472),
         audit("prior_541_critical23_v1","native541-critical23-input-002",138),
         audit("corrected_547_v2","native547-critical23-sweep-input-002",1472)]
g=results[2]
if g["zero_endpoint_dF"] or g["wrong_or_zero_sign_endpoint_dF"] or g["max_full_F_target_error"]>2e-15 or g["accepted_state_word_change_rows"] or g["reference_center_mismatch_rows"] or g["max_reference_realized_F_mismatch"]!=0:
    raise AssertionError("corrected v2 independent check failed")
report={"schema":"numi.cardboard.native547-critical23-input-audit.v1",
 "scope":"CPU-only input construction/admission check. Reconstruct serialized F from FP32 high/low words and compare with intended perturbation from immutable source center. No GPU/physical solve.",
 "source_center_input":{"path":str(CENTER_PATH),"sha256":sha(CENTER_PATH),"source_capture_sha256":CENTER["source_capture_sha256"],"compiled_package_sha256":CENTER["compiled_package_sha256"]},
 "results":results,
 "finding":{"superseded_547_v1":"INVALID_MUTATING_ENDPOINT_BUILDER: NumPy float32 endpoint high/low views aliased the reused center array; accompanying reference was generated from the same mutated state.",
 "prior_541_critical23_v1":"FD_ENDPOINTS_INVALID_FROM_SAME_ALIAS: the distinct 3456-row center replay remains separate; these 92 endpoints are not FD evidence.",
 "corrected_547_v2":"INPUT_SEMANTICS_PASS: copied endpoint split buffers and source-center immutability check. All endpoints are nonzero, correctly signed, and within 2e-15 of intended F."},
 "limits":["This is input and CPU-admission evidence only, not native material qualification.",
 "The local constitutive law is rate-independent; stress secants use realized delta-F only. Fdot is retained as physical velocity-gradient relation; FP32 rounding is separately reported.",
 "Smoothed yield/multiplier signs do not prove a discrete branch crossing.",
 "Native stress/tangent gates, finite-difference noise, and full coupon qualification remain open."],
 "audit_script_sha256":sha(__file__),
 "cpu_admission_v2":{"stdout_path":str(A/"native547-critical23-sweep-input-002/admission-v2.stdout"),
 "stdout_sha256":sha(A/"native547-critical23-sweep-input-002/admission-v2.stdout"),
 "stderr_sha256":sha(A/"native547-critical23-sweep-input-002/admission-v2.stderr"),"native_gpu_executed":False}}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
(OUT/"README.md").write_text("""# 547 critical23 endpoint input audit

The audit compares each serialized row to the immutable center row from native541-center-input-001/input.json, not merely to its accompanying reference file. It reconstructs F from exact FP32 high/low words, checks center F plus or minus 2^-p H, validates accepted-history words, and reports FP32 rate-gradient rounding separately.

The superseded 547 v1 and earlier 541 critical23 batch are retained as invalid finite-difference endpoint inputs due to aliased NumPy FP32 views. The separate 541 center-only native replay remains usable. Corrected 547 v2 passed CPU admission; this is input/admission evidence, not native material qualification.

Reproduce on macmini with:
 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 audit.py

The script expects the bound external inputs described in report.json.
""")
(OUT/"audit.py").write_text(pathlib.Path(__file__).read_text())
