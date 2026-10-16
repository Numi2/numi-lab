#!/usr/bin/env python3
import hashlib, importlib.util, json, pathlib, sys
import numpy as np

Q=pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
OUT=Q/"analysis/native497-analytic-warm-cgls-action-004"
OLD=Q/"analysis/native497-analytic-warm-cgls-action-003"
TOOLS=Q/"snapshots/source-portable-paper-reference-185/matter/tools"
RUNTIME=Q/"snapshots/source-portable-paper-reference-185/matter/src/runtime.mm"
KERNEL=Q/"snapshots/source-portable-paper-reference-185/matter/src/metal/base_cgls.metalinc"
if OUT.exists() and any(p.name != "seal_addendum.py" for p in OUT.iterdir()):
    raise SystemExit("output namespace is not fresh")
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def rowstats(expected,actual):
    e=np.asarray(expected,dtype=np.float64).reshape(-1,3)
    a=np.asarray(actual,dtype=np.float64).reshape(-1,3)
    d=a-e
    ne=np.linalg.norm(e,axis=1); nd=np.linalg.norm(d,axis=1)
    nz=ne>0
    ratios=np.full(len(e),np.nan); np.divide(nd,ne,out=ratios,where=nz)
    return {
      "rows":len(e),"global_expected_l2":float(np.linalg.norm(e)),
      "global_actual_l2":float(np.linalg.norm(a)),
      "global_absolute_difference_l2":float(np.linalg.norm(d)),
      "global_relative_l2":float(np.linalg.norm(d)/max(np.linalg.norm(e),1e-300)),
      "max_row_absolute_l2":float(nd.max()),
      "max_row_expected_l2":float(ne.max()),
      "max_row_relative_l2_nonzero_expected":float(np.nanmax(ratios)) if nz.any() else None,
      "worst_row":int(np.nanargmax(ratios)) if nz.any() else None
    }

sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location("action_audit",OLD/"audit.py")
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
prior_report=json.loads((OLD/"report.json").read_text())
if sha(OLD/"report.json")!="b6ab00d8d2eb62625fbc65878306f474213d1dd3ccd8ab5dec7d534258e91f1f":
    raise SystemExit("prior report hash changed")
meta, nodes, tets=a.parse_capture()
cg=a.read_cgls()
z=np.load(OLD/"arrays.npz",allow_pickle=False)
ids=z["free_node_ids"].astype(np.int64)
op=a.csr_load(a.OP/"operator.operator.npz")
x_capture=cg["x"][:1764].reshape(2,882,4)[0,:,:3][ids].astype(np.float64).reshape(-1)
x_report=z["warm_total_direction_free_xyz"].astype(np.float64).reshape(-1)
if not np.array_equal(x_capture,x_report): raise SystemExit("captured X differs from report vector")
x=x_capture
x0=np.asarray([np.asarray(nodes[int(i)]["correction_delta_v_m_s"],np.float32) for i in ids],dtype=np.float64).reshape(-1)
b=cg["b"][:1764].reshape(2,882,4)[0,:,:3][ids].astype(np.float64).reshape(-1)
r=cg["r"][:1764].reshape(2,882,4)[0,:,:3][ids].astype(np.float64).reshape(-1)
q=cg["Ax"][:1764].reshape(2,882,4)[0,:,:3][ids].astype(np.float64).reshape(-1)
ax=a.csr_mv(op,x); ax0=a.csr_mv(op,x0); acorr=a.csr_mv(op,x-x0)
rhs_res=b-r; reconstructed=ax0+rhs_res
runtime=RUNTIME.read_text().splitlines(); kernel=KERNEL.read_text().splitlines()
def matching(lines,needle):
    return [i+1 for i,s in enumerate(lines) if needle in s]
report={
 "schema":"numi.cardboard.native497-cgls-action-capture-erratum.v1",
 "evidence_class":"CPU-only source-bound interpretation and arithmetic audit; no native rerun",
 "source_case":{"root":20,"newton_iteration":63,"environment":0,"node_count":882,"tet_count":1920,"free_dof_count":2538,
   "world_fingerprint":meta["world_fingerprint_hex"],"execution_fingerprint":meta["execution_fingerprint_hex"]},
 "corrected_capture_semantics":{
   "x_segment":"Version-3 CGLS snapshot X is the total candidate x0+S*y after strict residual comparison; it matches the report warm_total_direction_free_xyz exactly after widening captured FP32 words.",
   "x0_segment":"Current Newton FGMRES femSolution, read from operator-FD node field correction_delta_v_m_s; it is not cumulative node delta_velocity_m_s.",
   "b_segment":"Fresh residual RHS b-A*x0 written by seed_residual for warm-start refinement.",
   "r_segment":"CGLS recurrence residual before final total-candidate action check.",
   "Ax_segment":"Fresh production cached-operator action A*X: runtime applies physical X then dispatches the mode-4 action to the buffer at snapshot segment Ax before taking the selected snapshot.",
   "prior_report_fields_corrected":[
      "input_layer.initial_FGMRES_correction_source incorrectly named delta_velocity_m_s; actual source is correction_delta_v_m_s.",
      "captured_q_workspace_vs_CSR_A_times_X_not_comparable incorrectly called the captured Ax segment a last-Ap workspace; it is the final A*X action.",
      "arrays key CGLS_last_q_Ap_workspace is a legacy mislabel for the captured Ax segment."
   ]
 },
 "capture_to_source_binding":{
   "runtime_sha256":sha(RUNTIME),"kernel_sha256":sha(KERNEL),
   "runtime_seed_residual_binding_lines":matching(runtime,"nm_fgmres_base_cgls_seed_residual"),
   "runtime_final_total_action_lines":matching(runtime,"action(4u, state.femBaseCGLSX, state.femBaseCGLSQ)"),
   "runtime_snapshot_encoder_call_lines":matching(runtime,"encodeBaseCauchyDirectionTrace(true)") + matching(runtime,"encodeBaseCauchyDirectionTrace()"),
   "runtime_snapshot_source_order_lines":matching(runtime,"state.femBaseCGLSX, state.femBaseCGLSR, state.femBaseCGLSS"),
   "kernel_seed_residual_binding_lines":matching(kernel,"kernel void nm_fgmres_base_cgls_seed_residual"),
   "prior_report_sha256":sha(OLD/"report.json"),
   "prior_arrays_sha256":sha(OLD/"arrays.npz"),
   "cgls_capture_sha256":sha(a.CG),
   "operator_manifest_sha256":sha(a.OP/"manifest.json")
 },
 "exact_state_and_action_metrics":{
   "x_total_l2_m_s":float(np.linalg.norm(x_capture)),
   "x0_current_fgmres_l2_m_s":float(np.linalg.norm(x0)),
   "x_total_minus_x0_l2_m_s":float(np.linalg.norm(x_capture-x0)),
   "captured_B_l2_Ns":float(np.linalg.norm(b)),
   "captured_R_l2_Ns":float(np.linalg.norm(r)),
   "captured_native_A_times_X_l2_Ns":float(np.linalg.norm(q)),
   "captured_native_A_times_X_vs_same_generation_fp64_csr_A_times_X":rowstats(ax.reshape(-1,3),q.reshape(-1,3)),
   "fp64_csr_A_times_total_minus_x0_vs_captured_B_minus_R":rowstats(rhs_res.reshape(-1,3),acorr.reshape(-1,3)),
   "captured_native_A_times_X_vs_fp64_csr_Ax0_plus_B_minus_R":rowstats(reconstructed.reshape(-1,3),q.reshape(-1,3)),
   "fp64_csr_A_times_total_vs_fp64_csr_Ax0_plus_B_minus_R":rowstats(ax.reshape(-1,3),reconstructed.reshape(-1,3)),
   "limitations":"The capture does not retain the native A*x0 vector before seed_residual. Therefore this residual/action gap cannot be separated into native FP32 action error on x0 versus recurrence/update/reduction drift. Comparisons use a same-generation CSR reconstructed from captured FP32 operator coefficients widened to FP64; they are not an independent exact physical operator."
 },
 "retained_material_tet_nodal_findings":{
   "paper_center_strict_projection":prior_report["strict_projection"],
   "paper_center_stress_vs_native_capture":prior_report["material_layer"]["center_stress_strict_vs_captured"],
   "strict_material_directional_tangent_vs_captured_fp32_cached_matrix":prior_report["material_layer"]["strict_directional_P_action_vs_widened_FP32_cached_matrix"],
   "strict_full_tet_frame_nodal_action_vs_same_generation_cached_csr_Ax":prior_report["assembled_action"]["strict_full_action_vs_exact_cached_operator_A_times_X"],
   "mass_action_vs_mass_csr":prior_report["assembled_action"]["mass_strict_vs_exact_operator_mass_CSR"],
   "cached_tangent_tet_action_vs_dt2_stiffness_csr":prior_report["assembled_action"]["cached_tangent_assembled_vs_exact_dt2_stiffness_CSR"],
   "strict_tangent_tet_action_vs_dt2_stiffness_csr":prior_report["assembled_action"]["strict_tangent_assembled_vs_exact_dt2_stiffness_CSR"],
   "first_layer_interpretation":"No material center stress mismatch is indicated. The largest resolved difference in this independent smooth action chain is the strict FP64 analytic tangent versus captured widened FP32 cached tangent/operator (global assembled action relative 2.9851e-5; max row relative 4.0331e-4). Mass reconstruction is exact against its captured CSR; cached tet/frame/scatter agrees with stiffness CSR at 1.19e-12. No native action was replayed."
 },
 "limits":[
   "Selected environment 0 at one captured Newton linearization only; no full-cycle or full-coupon qualification.",
   "No native GPU replay was run.",
   "No zero-direction absolute noise floor is inferred.",
   "The native recurrence/action identity is only bounded by available captures; native A*x0 was not separately retained.",
   "Strict reference versus cached FP32 tangent is descriptive and below aggregate 1e-3, but this is not a global calibration or all-direction proof."
 ]
}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
(OUT/"README.md").write_text("""# Native497 warm-CGLS action audit addendum

This addendum corrects the interpretation of the unchanged native497-analytic-warm-cgls-action-003 payload. Source185 confirms captured Ax is the final fresh action on the total candidate X, not the last Arnoldi/CG action workspace. It also binds x0 to correction_delta_v_m_s (the current Newton femSolution), not cumulative delta_velocity_m_s.

The strict material/tet/frame/nodal action results are retained verbatim from 003. Against the same-generation captured FP32-cache CSR, strict mass is exact and the cached tangent assembly agrees with stiffness at 1.19e-12 relative. The strict FP64 analytic tangent differs from that cached tangent by 2.916e-5 globally at stiffness and the total assembled action differs by 2.985e-5 globally (max nodal row relative 4.033e-4). Center stress vs captured native readback differs 2.39e-8 globally. These are CPU reconstruction results, not a native action replay.

The sidecar identity A*x = A*x0 + B - R misses by 3.77e-9 Ns absolute (1.999e-4 relative to captured native A*x). Same-generation CSR checks show A*(X-x0) differs from B-R by 3.625e-9 Ns, 1.743e-2 relative to that small residual. Because the capture did not retain the native A*x0 vector, this cannot isolate native FP32 action error from CGLS recurrence/update/reduction drift. It remains an unresolved solver-side discrepancy.

Run python3 seal_addendum.py with the source185 task venv and existing source/capture inputs available. The script verifies prior report identity and records the bound source lines and derived action checks; it does not alter the prior artifact.
""")
inv={}
for p in sorted(OUT.iterdir()):
 if p.is_file() and p.name!="manifest.json":
  inv[p.name]={"bytes":p.stat().st_size,"sha256":sha(p)}
(OUT/"manifest.json").write_text(json.dumps({"schema":"numi.cardboard.analysis-package.v1","package":"native497-analytic-warm-cgls-action-004","scope":"source-bound interpretation correction plus exact CGLS residual/action identity audit","files":inv},indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(OUT),"manifest_sha256":sha(OUT/"manifest.json"),"report_sha256":sha(OUT/"report.json"),"metrics":report["exact_state_and_action_metrics"],"material":report["retained_material_tet_nodal_findings"]["first_layer_interpretation"]},indent=2))

