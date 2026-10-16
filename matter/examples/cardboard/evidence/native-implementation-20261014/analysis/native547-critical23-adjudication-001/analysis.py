#!/usr/bin/env python3
import hashlib, importlib.util, json, pathlib, struct, sys
import numpy as np

Q=pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
A=Q/"analysis"
PK=A/"native547-critical23-sweep-input-002"
RUN=Q/"native/critical23-power-sweep-547"
NATIVE=RUN/"output/paper-points-exact_package_budget32.jsonl"
COMPILED=Q/"solver/control17-root17-newton34-replay-531/output/compiled.nmatterpack"
HELPER=A/"native541-generic-package-batch-instrument-001/generic-batch"
SOURCE=Q/"snapshots/source-cohesive-quadrature-area-205"
REFMODULE=Q/"analysis/native531-stable-green-strain-reference-017-001/reference/cardboard_bilinear_fp64_reference.py"
OUT=A/"native547-critical23-adjudication-001"
if (OUT/"manifest.json").exists(): raise SystemExit("sealed adjudication; refusing overwrite")
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def f32(x): return np.asarray(x,dtype=np.float32)
def frombits(a): return np.asarray([int(x,16) for x in a],dtype=np.uint32).view(np.float32).astype(np.float64)
spec=importlib.util.spec_from_file_location("stable_green_reference",REFMODULE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
inp=json.loads((PK/"input.json").read_text())
refs=[json.loads(x) for x in (PK/"stable-fp64-reference.jsonl").read_text().splitlines()]
lines=[json.loads(x) for x in NATIVE.read_text().splitlines()]
header,end=lines[0],lines[-1]; pts=lines[1:-1]
assert header.get("schema")=="numi.cardboard.bilinear15-capture-diagnostic.v1"
assert header.get("record")=="header" and end.get("record")=="end"
assert header["point_input_sha256"]==sha(PK/"input.json")
assert header["record_count"]==len(inp["rows"])==len(refs)==len(pts)==1472
assert end["rows"]==1472
assert header["base_snapshot_manifest_sha256"]=="c77675e7aedc7f14c93c8bd7ea9e0a1e18bf162a68cf7e4022c43c34c6d141f2"
assert sha(COMPILED)=="7b4df5b67bacfd579e3ce4b6776457f581cfb3240cbc70552b5d5c716f987be1"
assert sha(HELPER)=="7764e4c20c2393d5ebfc3f100f29e851b9090e1999418aad86cbcaf69fb7e3c5"
center_errors=[]; stress=[]; tangents=[]; state_errors=[]; invalid=[]; direct_path=[]; branch_rows=[]
for i,(row,ref,n) in enumerate(zip(inp["rows"],refs,pts)):
    assert ref["index"]==i and n["record"]=="point"
    assert int(row["tet_id"])==int(ref["tet"])==int(n["tet_id"])
    assert int(row["material_index"])==int(ref["material"])==int(n["material_index"])
    assert int(row["environment"])==int(ref["env"])==int(n["environment"])
    bits=n["output_f32_bits"]
    if len(bits)!=65: raise AssertionError(f"row {i}: expected 65 float words")
    vals=frombits(bits)
    direct=vals[8:17]; tpath=vals[17:26]; tan=vals[26:35]
    state=vals[35:50]+vals[50:65]
    # Exported JSON convenience arrays must encode the same FP32 words.
    if not np.array_equal(f32(n["direct_stress_pa"]).view(np.uint32),f32(direct).view(np.uint32)):
        raise AssertionError(f"row {i}: explicit direct stress vs bit payload mismatch")
    if not np.array_equal(f32(n["tangent_path_stress_pa"]).view(np.uint32),f32(tpath).view(np.uint32)):
        raise AssertionError(f"row {i}: explicit tangent-path stress vs bit payload mismatch")
    if not np.array_equal(f32(n["tangent_pa_per_s"]).view(np.uint32),f32(tan).view(np.uint32)):
        raise AssertionError(f"row {i}: explicit tangent vs bit payload mismatch")
    if not (n["stress_valid"] and n["tangent_valid"] and n["tangent_projection_converged"]):
        invalid.append({"index":i,"tet":ref["tet"],"label":ref["label"],"power":ref["power"],"sign":ref["sign"],
          "stress_valid":n["stress_valid"],"tangent_valid":n["tangent_valid"],
          "projection_converged":n["tangent_projection_converged"],"failure":n["projection_failure"]})
    refstress=np.asarray(ref.get("stress",np.zeros(9)),dtype=np.float64).reshape(9)
    reftan=np.asarray(ref.get("tangent",np.zeros(9)),dtype=np.float64).reshape(9)
    refstate=np.asarray(ref.get("state",np.zeros(15)),dtype=np.float64).reshape(15)
    sa=float(np.linalg.norm(direct-refstress)); sr=sa/max(float(np.linalg.norm(refstress)),1e-300)
    ta=float(np.linalg.norm(tan-reftan)); tr=ta/max(float(np.linalg.norm(reftan)),1e-300)
    st=float(np.linalg.norm(state-refstate))
    dp=float(np.max(np.abs(direct-tpath)))
    stress.append({"index":i,"label":ref["label"],"tet":ref["tet"],"material":ref["material"],
       "basis_column":ref["basis_column"],"power":ref["power"],"sign":ref["sign"],
       "native_stress_abs_l2":sa,"native_stress_rel_l2":sr,
       "native_stress_max_abs_component":float(np.max(np.abs(direct-refstress))),
       "strict_stress_norm":float(np.linalg.norm(refstress)),"native_stress_norm":float(np.linalg.norm(direct)),
       "direct_vs_tangent_path_max_abs":dp})
    tangents.append({"index":i,"label":ref["label"],"tet":ref["tet"],"basis_column":ref["basis_column"],
       "power":ref["power"],"sign":ref["sign"],"native_tangent_abs_l2":ta,"native_tangent_rel_l2":tr,
       "native_tangent_max_abs_component":float(np.max(np.abs(tan-reftan))),
       "strict_tangent_norm":float(np.linalg.norm(reftan)),"native_tangent_norm":float(np.linalg.norm(tan)),
       "native_tangent_gate_1e-3":bool(tr<=1e-3)})
    state_errors.append({"index":i,"label":ref["label"],"tet":ref["tet"],"basis_column":ref["basis_column"],
       "power":ref["power"],"sign":ref["sign"],"native_vs_strict_state_abs_l2":st,
       "native_projection_residual":float(n["projection_residual"]),"projection_iterations":int(n["projection_iterations"])})
    direct_path.append(dp)
    if ref["strict_valid"]:
        p=f32(inp["parameters_by_material"][int(ref["material"])]).astype(np.float64)[None,:]
        old=f32(row["accepted_state_f32"]).astype(np.float64)[None,:]
        bs=base.paper_branch(base,p,old,state[None,:])
        dl=float(bs["plastic_multiplier_increment"][0]); y=float(bs["yield_function_residual_value"][0])
        branch_rows.append({"index":i,"label":ref["label"],"tet":ref["tet"],"basis_column":ref["basis_column"],
          "power":ref["power"],"sign":ref["sign"],"strict_branch_indicator":ref.get("branch"),
          "native_smoothed_dl_sign":int(np.sign(dl)),"native_smoothed_y_sign":int(np.sign(y)),
          "native_fb_den_gt_1e-14":bool(np.sqrt(dl*dl+y*y+float(p[0,26])**2)+dl+y>1e-14),
          "native_dl":dl,"native_y":y,"strict_projection_ok":bool(ref["strict_valid"])})
# Duplicate center rows are direct same-input repeatability only.
repeats=[]
for pair_start in range(0,len(pts),64):
    a,b=pts[pair_start],pts[pair_start+1]
    repeats.append({"tet":a["tet_id"],"material":a["material_index"],
       "bit_identical_65_words":a["output_f32_bits"]==b["output_f32_bits"],
       "different_slots":[j for j,(x,y) in enumerate(zip(a["output_f32_bits"],b["output_f32_bits"])) if x!=y]})
# Realized centered secants use actual serialized FP32 F sums, never nominal epsilon.
idx_by={}
for i,r in enumerate(refs):
    if r["label"]=="endpoint": idx_by.setdefault((int(r["tet"]),int(r["basis_column"]),int(r["power"])),{})[int(r["sign"])]=i
center_idx={(int(r["tet"]),int(r["basis_column"])):i for i,r in enumerate(refs) if r["label"]=="center"}
fd=[]
for (tet,col,power),g in sorted(idx_by.items()):
    ip,im=g[1],g[-1]; rp,rm=refs[ip],refs[im]; np_,nm=pts[ip],pts[im]
    wp=f32(inp["rows"][ip]["packed_input_f32"]); wm=f32(inp["rows"][im]["packed_input_f32"])
    fp=(wp[:9].astype(np.float64)+wp[9:18].astype(np.float64)).reshape(9)[col]
    fm=(wm[:9].astype(np.float64)+wm[9:18].astype(np.float64)).reshape(9)[col]
    sep=float(fp-fm); eps_eff=0.5*sep
    sp=frombits(np_["output_f32_bits"])[8:17]; sm=frombits(nm["output_f32_bits"])[8:17]
    srp=np.asarray(rp.get("stress",np.zeros(9)),dtype=np.float64).reshape(9)
    srm=np.asarray(rm.get("stress",np.zeros(9)),dtype=np.float64).reshape(9)
    center_ref=refs[center_idx[(tet,col)]]
    tc=np.asarray(center_ref.get("tangent",np.zeros(9)),dtype=np.float64).reshape(9)
    ncenter=frombits(pts[center_idx[(tet,col)]]["output_f32_bits"])[26:35]
    nslope=(sp-sm)/sep if sep else np.full(9,np.nan)
    rslope=(srp-srm)/sep if sep else np.full(9,np.nan)
    signal=float(np.linalg.norm(sp-sm))
    endpoint_error=float(np.linalg.norm(sp-srp)+np.linalg.norm(sm-srm))
    strict_valid=bool(rp["strict_valid"] and rm["strict_valid"])
    native_valid=bool(np_["stress_valid"] and nm["stress_valid"])
    same_smooth=(rp.get("branch")==rm.get("branch")==center_ref.get("branch"))
    bmap={x["index"]:(x["native_smoothed_dl_sign"],x["native_smoothed_y_sign"],x["native_fb_den_gt_1e-14"]) for x in branch_rows}
    same_native_smooth=(bmap.get(ip)==bmap.get(im)==bmap.get(center_idx[(tet,col)]))
    nrel=float(np.linalg.norm(nslope-ncenter)/max(np.linalg.norm(ncenter),1e-300)) if sep else None
    rrel=float(np.linalg.norm(rslope-tc)/max(np.linalg.norm(tc),1e-300)) if sep else None
    native_vs_ref_tangent=float(np.linalg.norm(ncenter-tc)/max(np.linalg.norm(tc),1e-300))
    resolvable=bool(signal>endpoint_error)
    fd.append({"tet":tet,"basis_column":col,"power":power,"nominal_epsilon":float(rp["epsilon"]),
      "realized_plus_minus_separation":sep,"realized_half_separation":eps_eff,
      "both_strict_projections_valid":strict_valid,"both_native_stresses_valid":native_valid,
       "same_smooth_indicator_tuple":same_smooth,"same_native_smoothed_indicator_tuple":same_native_smooth,
      "strict_centered_secant_rel_error_vs_strict_analytic":rrel,
      "native_centered_secant_rel_error_vs_native_center_tangent":nrel,
      "native_centered_secant_rel_error_vs_strict_center_tangent":float(np.linalg.norm(nslope-tc)/max(np.linalg.norm(tc),1e-300)) if sep else None,
      "strict_center_analytic_tangent_gate_1e-6":bool(rrel is not None and rrel<=1e-6),
      "native_center_tangent_vs_strict_gate_1e-3":bool(native_vs_ref_tangent<=1e-3),
      "native_signal_l2":signal,"measured_native_to_strict_endpoint_error_sum_l2":endpoint_error,
      "native_signal_exceeds_endpoint_discrepancy_bound":resolvable,
      "interpretation":"measured endpoint discrepancy bound, not a repeatability or universal absolute noise floor" if not resolvable else "signal exceeds this pair's measured endpoint discrepancy sum"})
# Group adjacent powers and identify windows passing strict analytic FD only when both adjacent powers, signs valid, and smooth indicators stable.
fd_by={}
for x in fd: fd_by.setdefault((x["tet"],x["basis_column"]),{})[x["power"]]=x
windows=[]
for key,d in sorted(fd_by.items()):
    passes=[]
    for p in sorted(d):
        if p+1 in d:
            a,b=d[p],d[p+1]
            if (a["both_strict_projections_valid"] and b["both_strict_projections_valid"] and
                a["same_smooth_indicator_tuple"] and b["same_smooth_indicator_tuple"] and
                a["strict_center_analytic_tangent_gate_1e-6"] and b["strict_center_analytic_tangent_gate_1e-6"]):
                passes.append([p,p+1])
    native_passes=[]; joint_passes=[]
    for p in sorted(d):
        if p+1 not in d: continue
        a,b=d[p],d[p+1]
        native_eligible=lambda x: (x["both_native_stresses_valid"] and x["both_strict_projections_valid"] and
          x["same_smooth_indicator_tuple"] and x["same_native_smoothed_indicator_tuple"] and
          x["native_signal_exceeds_endpoint_discrepancy_bound"] and
          x["native_center_tangent_vs_strict_gate_1e-3"] and
          x["native_centered_secant_rel_error_vs_strict_center_tangent"] is not None and
          x["native_centered_secant_rel_error_vs_strict_center_tangent"]<=1e-3)
        if native_eligible(a) and native_eligible(b): native_passes.append([p,p+1])
        if ([p,p+1] in passes) and native_eligible(a) and native_eligible(b): joint_passes.append([p,p+1])
    windows.append({"tet":key[0],"basis_column":key[1],"adjacent_strict_1e-6_windows":passes,
       "any_adjacent_strict_window":bool(passes),"adjacent_native_resolved_1e-3_windows":native_passes,
       "adjacent_joint_native_and_strict_windows":joint_passes})
def stats(rows,k):
    vals=[x[k] for x in rows]
    return {"count":len(vals),"max":max(vals,default=None),"median":float(np.median(vals)) if vals else None}
center_rows=[x for x in stress if x["label"]=="center"]
center_tans=[x for x in tangents if x["label"]=="center"]
all_stress=[x for x in stress]
all_tans=[x for x in tangents]
summary={
 "schema":"numi.cardboard.native547-critical23-adjudication.v1",
 "bindings":{"input_sha256":sha(PK/"input.json"),"pair_selection_sha256":sha(PK/"pair-selection.json"),
   "strict_reference_sha256":sha(PK/"stable-fp64-reference.jsonl"),"native_jsonl_sha256":sha(NATIVE),
   "compiled_package_sha256":sha(COMPILED),"helper_sha256":sha(HELPER),
   "run_receipt_sha256":sha(RUN/"receipt.json"),"input_audit_manifest_sha256":sha(A/"native547-critical23-sweep-input-audit-008/manifest.json"),
   "runner_binding_sha256":sha(A/"native547-root-corrected-runner-002/runner-binding.json")},
 "native_run":{"record_count":header["record_count"],"native_rows":len(pts),"native_status":end["status"],
   "stress_valid":end["direct_stress_valid_count"],"tangent_valid":end["tangent_valid_count"],
   "tangent_projection_converged":end["tangent_projection_converged_count"],
   "physical_transactions":json.loads((RUN/"receipt.json").read_text())["physical_transactions"],
   "compiled_local_iterations_liner":header["compiled_local_iterations_liner"],
   "compiled_local_iterations_medium":header["compiled_local_iterations_medium"],
   "invalid_rows":invalid},
 "center_repeatability":{"pairs":len(repeats),"bit_identical_pairs":sum(r["bit_identical_65_words"] for r in repeats),
   "all":repeats,"meaning":"repeatability only; not an absolute arithmetic noise floor"},
 "native_vs_strict_FP64":{"scope":"23 selected material/basis pairs, 46 center rows and all 1426 endpoint rows; stress/tangent normalized L2 is per captured direction, not a full 9x9 matrix gate.",
   "center_stress":{"rows":len(center_rows),"max_relative_l2":max((x["native_stress_rel_l2"] for x in center_rows),default=None),
      "max_absolute_l2":max((x["native_stress_abs_l2"] for x in center_rows),default=None),
      "max_abs_component":max((x["native_stress_max_abs_component"] for x in center_rows),default=None)},
   "center_tangent":{"rows":len(center_tans),"max_relative_l2":max((x["native_tangent_rel_l2"] for x in center_tans),default=None),
      "max_absolute_l2":max((x["native_tangent_abs_l2"] for x in center_tans),default=None),
      "over_1e-3_rows":sum(not x["native_tangent_gate_1e-3"] for x in center_tans),
      "worst_rows":sorted(center_tans,key=lambda x:x["native_tangent_rel_l2"],reverse=True)[:12]},
   "all_row_stress":{"rows":len(all_stress),"max_relative_l2":max((x["native_stress_rel_l2"] for x in all_stress),default=None),
      "max_absolute_l2":max((x["native_stress_abs_l2"] for x in all_stress),default=None),
      "max_abs_component":max((x["native_stress_max_abs_component"] for x in all_stress),default=None)},
   "all_row_tangent":{"rows":len(all_tans),"max_relative_l2":max((x["native_tangent_rel_l2"] for x in all_tans),default=None),
      "max_absolute_l2":max((x["native_tangent_abs_l2"] for x in all_tans),default=None),
      "over_1e-3_rows":sum(not x["native_tangent_gate_1e-3"] for x in all_tans),
      "worst_rows":sorted(all_tans,key=lambda x:x["native_tangent_rel_l2"],reverse=True)[:12]},
   "max_native_direct_vs_tangent_path_abs":max(direct_path,default=0.0),
   "max_native_projected_state_vs_strict_abs_l2":max((x["native_vs_strict_state_abs_l2"] for x in state_errors),default=None)},
 "realized_finite_differences":{"pair_power_rows":len(fd),"basis_pairs":len(windows),
    "strict_adjacent_1e-6_windows":sum(x["any_adjacent_strict_window"] for x in windows),
   "native_adjacent_resolved_1e-3_windows":sum(bool(x["adjacent_native_resolved_1e-3_windows"]) for x in windows),
   "joint_adjacent_native_and_strict_windows":sum(bool(x["adjacent_joint_native_and_strict_windows"]) for x in windows),
   "adjacent_windows":windows,
   "per_pair_power":fd},
 "branch_indicators":{"rows":len(branch_rows),"meaning":"smooth multiplier/yield/FB indicators only; not proof of discrete constitutive branch crossing",
   "native_vs_reference_smooth_tuple_mismatches":sum(1 for i,x in enumerate(branch_rows) if refs[i].get("branch")!=[x["native_smoothed_dl_sign"],x["native_smoothed_y_sign"],x["native_fb_den_gt_1e-14"]])},
 "limits":["The superseded 541 critical23 endpoint batch is invalid and was not used; this adjudication uses corrected 547 v2 input only.",
   "Only 23 selected tet/basis directions are present; this is not the full nine-direction tensor qualification.",
   "Center native tangent-vs-strict reference, strict-reference FD self-consistency, and native FD-vs-tangent/noise resolution are separate results.",
   "The local constitutive law is rate-independent: centered constitutive secants use realized deformation-gradient separation only; Fdot is preserved as the corresponding physical gradient input but is not averaged into the stress derivative.",
   "Same-run identical center repeats establish repeatability only. Pairwise endpoint discrepancy is reported as a measured signal bound, not a universal noise floor.",
   "Smoothed sign indicators are not discrete branch labels. No physical coupon, calibration, or full solver qualification is claimed."]
}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"report.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
with (OUT/"stress-rows.jsonl").open("w") as f:
    for x in stress: f.write(json.dumps(x,sort_keys=True,allow_nan=False)+"\n")
with (OUT/"tangent-rows.jsonl").open("w") as f:
    for x in tangents: f.write(json.dumps(x,sort_keys=True,allow_nan=False)+"\n")
with (OUT/"fd-rows.jsonl").open("w") as f:
    for x in fd: f.write(json.dumps(x,sort_keys=True,allow_nan=False)+"\n")
with (OUT/"state-rows.jsonl").open("w") as f:
    for x in state_errors: f.write(json.dumps(x,sort_keys=True,allow_nan=False)+"\n")
(OUT/"analysis.py").write_text(pathlib.Path(__file__).read_text())
print(json.dumps({"out":str(OUT),"report_sha256":sha(OUT/"report.json"),"native_rows":len(pts),
  "strict_projections":sum(bool(x["strict_valid"]) for x in refs),"native_valid":len(pts)-len(invalid),
  "center_stress_max_rel":summary["native_vs_strict_FP64"]["center_stress"]["max_relative_l2"],
  "center_tangent_max_rel":summary["native_vs_strict_FP64"]["center_tangent"]["max_relative_l2"],
  "center_tangent_over_1e3":summary["native_vs_strict_FP64"]["center_tangent"]["over_1e-3_rows"],
  "all_adjacent_strict1e6":summary["realized_finite_differences"]["strict_adjacent_1e-6_windows"],
  "pairs":len(windows)},indent=2))
