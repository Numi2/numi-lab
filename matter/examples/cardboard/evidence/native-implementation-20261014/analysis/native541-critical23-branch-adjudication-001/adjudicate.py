import hashlib,importlib.util,json,sys,math
from pathlib import Path
import numpy as np
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PK=Q/"analysis/native541-critical23-input-002"; OUT=Q/"native/critical23-material-replay-541/output/paper-points-exact_package_budget32.jsonl"
PKG=Q/"solver/control17-root17-newton34-replay-531/output/compiled.nmatterpack"; HELP=Q/"analysis/native541-generic-package-batch-instrument-001/generic-batch"
STABLE=Q/"analysis/native531-stable-green-strain-reference-017-001/reference/cardboard_bilinear_fp64_reference.py"; SRC=Q/"snapshots/source-cohesive-quadrature-area-205"; TD=SRC/"matter/tools"
sys.path.insert(0,str(TD)); import cardboard_compensated_fp64_reference as strict
sp=importlib.util.spec_from_file_location("base",STABLE); base=importlib.util.module_from_spec(sp); sp.loader.exec_module(base)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(a): return np.asarray(a,dtype=np.float32).astype(np.float64)
def bits(a): return np.asarray([int(x,16) for x in a],dtype=np.uint32).view(np.float32).astype(np.float64)
inp=json.loads((PK/"input.json").read_text()); refs=[json.loads(x) for x in (PK/"stable-fp64-reference.jsonl").read_text().splitlines()]
raw=[json.loads(x) for x in OUT.read_text().splitlines()]; head,pts,end=raw[0],raw[1:-1],raw[-1]
assert len(refs)==len(pts)==138 and head["record_count"]==end["rows"]==138
assert head["point_input_sha256"]==sha(PK/"input.json")
assert sha(HELP)=="7764e4c20c2393d5ebfc3f100f29e851b9090e1999418aad86cbcaf69fb7e3c5"
assert sha(PKG)=="7b4df5b67bacfd579e3ce4b6776457f581cfb3240cbc70552b5d5c716f987be1"
def calc(p,o,z):
 b=base.paper_branch(base,p,o,z); dl=float(b["plastic_multiplier_increment"][0]); y=float(b["yield_function_residual_value"][0]); eps=float(p[0,26])
 root=math.sqrt(dl*dl+y*y+eps*eps); den=root+dl+y
 q,flow=base.hill(p,z[:,6:12]); flow=flow[0]; grad=np.zeros(15)
 grad[6:9]=-flow[:3];grad[9:12]=-2*flow[3:]
 grad[13]=float(p[0,17]*p[0,19]*p[0,21]/p[0,10]);grad[14]=float(p[0,18]*p[0,20]*p[0,22]/p[0,10])
 sn=z[0,6:9];width=float(p[0,23]);pos=.5*(sn+np.sqrt(sn*sn+width*width))
 return dict(dl=dl,y=y,eps=eps,den=den,q=float(q[0]),flow=flow,grad=grad,
  dl_pos=dl>0,y_pos=y>0,den_branch=den>1e-14,normal=sn.copy(),normal_sign=np.sign(sn).astype(int),
  normal_over_width=sn/width,positive_weights=pos)
details=[]; group={}
for i,(r,n) in enumerate(zip(refs,pts)):
 assert n["record"]=="point" and n["environment"]==0 and n["tet_id"]==r["tet"] and n["material_index"]==r["material"]
 v=bits(n["output_f32_bits"]); zn=(v[35:50]+v[50:65])[None,:]; zr=np.asarray(r["state"],float)[None,:]
 p=f32(inp["parameters_by_material"][r["material"]])[None,:]; old=f32(inp["rows"][i]["accepted_state_f32"])[None,:]
 mn=calc(p,old,zn);mr=calc(p,old,zr); dz=zn[0]-zr[0]; dy=mn["y"]-mr["y"]; pred=float(mr["grad"]@dz)
 row=dict(i=i,tet=r["tet"],material=r["material"],label=r["label"],basis=r["basis_column"],power=r["power"],sign=r["sign"],
  native_dl=mn["dl"],ref_dl=mr["dl"],native_y=mn["y"],ref_y=mr["y"],fb_eps=mn["eps"],
  native_y_over_eps=mn["y"]/mn["eps"],ref_y_over_eps=mr["y"]/mr["eps"],
  native_dl_over_eps=mn["dl"]/mn["eps"],ref_dl_over_eps=mr["dl"]/mr["eps"],
  native_den=mn["den"],ref_den=mr["den"],native_den_over_eps=mn["den"]/mn["eps"],ref_den_over_eps=mr["den"]/mr["eps"],
  native_dl_positive=mn["dl_pos"],ref_dl_positive=mr["dl_pos"],native_y_positive=mn["y_pos"],ref_y_positive=mr["y_pos"],
  native_den_gt_1e_14=mn["den_branch"],ref_den_gt_1e_14=mr["den_branch"],
  native_normal_sign=mn["normal_sign"].tolist(),ref_normal_sign=mr["normal_sign"].tolist(),
  native_normal_over_width=mn["normal_over_width"].tolist(),ref_normal_over_width=mr["normal_over_width"].tolist(),
  native_smooth_positive_weights=mn["positive_weights"].tolist(),ref_smooth_positive_weights=mr["positive_weights"].tolist(),
  state_error_l2=float(np.linalg.norm(dz)),delta_y=dy,delta_y_linear_from_state=pred,delta_y_nonlinear_remainder=dy-pred,
  native_valid=bool(n["stress_valid"] and n["tangent_valid"] and n["tangent_projection_converged"]),
  stress_rel=float(np.linalg.norm(v[8:17]-np.asarray(r["stress"]).reshape(9))/max(np.linalg.norm(r["stress"]),1e-300)),
  tangent_rel=float(np.linalg.norm(v[26:35]-np.asarray(r["tangent"]).reshape(9))/max(np.linalg.norm(r["tangent"]),1e-300)))
 details.append(row); group.setdefault((r["tet"],r["basis_column"]),[]).append(row)
ym=[r for r in details if r["native_y_positive"]!=r["ref_y_positive"]]
dlm=[r for r in details if r["native_dl_positive"]!=r["ref_dl_positive"]]
denm=[r for r in details if r["native_den_gt_1e_14"]!=r["ref_den_gt_1e_14"]]
nsm=[r for r in details if r["native_normal_sign"]!=r["ref_normal_sign"]]
const={}
for (tet,col),g in group.items():
 nf={(r["native_dl_positive"],r["native_y_positive"],r["native_den_gt_1e_14"]) for r in g}
 rf={(r["ref_dl_positive"],r["ref_y_positive"],r["ref_den_gt_1e_14"]) for r in g}
 const[f"{tet}:{col}"]={"native_flags":[list(x) for x in sorted(nf)],"ref_flags":[list(x) for x in sorted(rf)],
    "native_constant_over_six":len(nf)==1,"ref_constant_over_six":len(rf)==1}
oldr=json.loads((Q/"analysis/native541-critical23-adjudication-001/report.json").read_text())
s=oldr["native_vs_independent_stable_FP64"]; fd=oldr["endpoint_derivatives"]["all_pairs"]
src_material=SRC/"matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter"
src_common=SRC/"matter/src/metal/common.metalinc"
report=dict(schema="numi.cardboard.native541-critical23-branch-adjudication.v1",
 bindings=dict(input_sha256=sha(PK/"input.json"),reference_sha256=sha(PK/"stable-fp64-reference.jsonl"),native_jsonl_sha256=sha(OUT),
 package_sha256=sha(PKG),helper_sha256=sha(HELP),material_sha256=sha(src_material),common_metal_sha256=sha(src_common),
 source_capture_sha256="c9a538bd6357136130fc3d971c7d2f1e61fc52bdef2a0737ba00790bd34d8ec2",
 prior_adjudication_sha256=sha(Q/"analysis/native541-critical23-adjudication-001/report.json")),
 execution=dict(rows=len(pts),stress_valid=end["direct_stress_valid_count"],tangent_valid=end["tangent_valid_count"],
 projection_converged=end["tangent_projection_converged_count"],invalid_rows=[r["i"] for r in details if not r["native_valid"]],physical_fem_transactions=0),
 branch_indicators=dict(y_sign_mismatch_rows=len(ym),mismatch_tets=sorted(set(r["tet"] for r in ym)),y_sign_mismatch_details=ym,
 plastic_multiplier_positive_mismatch_rows=len(dlm),denominator_threshold_mismatch_rows=len(denm),normal_stress_sign_mismatch_rows=len(nsm),
 per_pair_constancy=const,
 interpretation="y sign is an indicator, not a constitutive branch. This material uses positive FB epsilon and a positive hardening smoothing width; its implicit residual has no hard y=0 or tension/compression split. The native generated bytecode evaluates the smooth authored equation. The FP64 reference's rationalized residual has a denominator>1e-14 evaluation choice; both policies choose that positive-denominator formula here, but it is not a native bytecode branch."),
 smooth_law=dict(fb_epsilon=float(np.float32(inp["parameters_by_material"][refs[0]["material"]][26])),
 hardening_smoothing_width=float(np.float32(inp["parameters_by_material"][refs[0]["material"]][23])),
 reference_quotient_cutoff=1e-14, equations="Haj-Ali medium .nmatter implicit residual; smooth positive-stress hardening weights"),
 measurements=dict(max_stress_abs_l2=s["max_stress_abs_l2"],max_stress_rel_l2=s["max_stress_rel_l2"],
 max_tangent_rel_l2=s["max_tangent_rel_l2"],max_state_abs_l2=s["max_projected_state_abs_l2"],
 selected_native_tangent_rows_le_1e_3=sum(r["tangent_rel"]<=1e-3 for r in details),selected_rows=len(details),
 fd_pairs=len(fd),fd_signal_exceeds_endpoint_error_sum=sum(r["signal_exceeds_endpoint_error_sum"] for r in fd),
 fd_signal_below_endpoint_error_sum=sum(not r["signal_exceeds_endpoint_error_sum"] for r in fd)),
 rows=details,
 limits=["Only 23 selected tet/basis pairs and two nearby epsilon powers each, not all 9 directions or all paper material rows.",
 "Same-run repeated output words are repeatability, not absolute noise floor. Endpoint error sums are measured discrepancies in this reference comparison, not a universal noise model.",
 "A y-sign mismatch with |y| much smaller than fb_epsilon does not establish a discontinuous constitutive branch crossing. The tangent and finite-difference gates remain distinct.",
 "These data are a bounded material replay, not calibration or coupon qualification."])
outdir=Q/"analysis/native541-critical23-branch-adjudication-001";outdir.mkdir(parents=True,exist_ok=True)
rp=outdir/"report.json";rp.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps(dict(report=str(rp),report_sha256=sha(rp),rows=len(details),invalid=len(report["execution"]["invalid_rows"]),
 y_mismatches=len(ym),tets=sorted(set(r["tet"] for r in ym)),dl_mismatches=len(dlm),den_threshold_mismatches=len(denm),
 normal_sign_mismatches=len(nsm),tangent_gate_rows=report["measurements"]["selected_native_tangent_rows_le_1e_3"],
 endpoint_signals_above_error_sum=report["measurements"]["fd_signal_exceeds_endpoint_error_sum"],
 endpoint_signals_below_error_sum=report["measurements"]["fd_signal_below_endpoint_error_sum"]),indent=2))

