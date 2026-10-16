import hashlib,importlib.util,json,sys,struct,warnings
from pathlib import Path
import numpy as np
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PK=Q/"analysis/native541-critical23-input-002"
OUT=Q/"native/critical23-material-replay-541/output/paper-points-exact_package_budget32.jsonl"
PKG=Q/"solver/control17-root17-newton34-replay-531/output/compiled.nmatterpack"
HELP=Q/"analysis/native541-generic-package-batch-instrument-001/generic-batch"
STABLE=Q/"analysis/native531-stable-green-strain-reference-017-001/reference/cardboard_bilinear_fp64_reference.py"
STRICTDIR=Q/"snapshots/source-cohesive-quadrature-area-205/matter/tools"
sys.path.insert(0,str(STRICTDIR))
import cardboard_compensated_fp64_reference as strict
spec=importlib.util.spec_from_file_location("stable_green_reference",STABLE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x):return np.asarray(x,dtype=np.float32)
def frombits(a):return np.asarray([int(x,16) for x in a],dtype=np.uint32).view(np.float32).astype(np.float64)
def ulpdist(a,b):
 def key(f):
  i=struct.unpack("<I",struct.pack("<f",float(np.float32(f))))[0]
  return (~i & 0xffffffff) if (i & 0x80000000) else (i|0x80000000)
 return abs(key(a)-key(b))
def sig(p,o,z):
 b=base.paper_branch(base,p,o,z);dl=b["plastic_multiplier_increment"];y=b["yield_function_residual_value"]
 den=np.sqrt(dl*dl+y*y+p[:,26]**2)+dl+y
 return [float(np.sign(dl[0])),float(np.sign(y[0])),bool(den[0]>1e-14)]
inp=json.loads((PK/"input.json").read_text())
refs=[json.loads(x) for x in (PK/"stable-fp64-reference.jsonl").read_text().splitlines()]
native=[json.loads(x) for x in OUT.read_text().splitlines()]
header=native[0];end=native[-1];pts=native[1:-1]
assert header["record_count"]==138 and len(pts)==138 and end["rows"]==138
assert header["point_input_sha256"]==sha(PK/"input.json")
assert header["base_snapshot_manifest_sha256"]=="c77675e7aedc7f14c93c8bd7ea9e0a1e18bf162a68cf7e4022c43c34c6d141f2"
assert sha(HELP)=="7764e4c20c2393d5ebfc3f100f29e851b9090e1999418aad86cbcaf69fb7e3c5"
assert sha(PKG)=="7b4df5b67bacfd579e3ce4b6776457f581cfb3240cbc70552b5d5c716f987be1"
bad=[];stress_errors=[];state_errors=[];tan_errors=[];ulp=[];branch_mismatch=[]
for i,(r,n) in enumerate(zip(refs,pts)):
 assert n["record"]=="point" and n["environment"]==0 and n["tet_id"]==r["tet"] and n["material_index"]==r["material"]
 bitsout=n["output_f32_bits"];assert len(bitsout)==65
 vals=frombits(bitsout)
 if not (n["stress_valid"] and n["tangent_valid"] and n["tangent_projection_converged"]):bad.append(i)
 direct=vals[8:17];tpath=vals[17:26];tan=vals[26:35]
 state=vals[35:50]+vals[50:65]
 refstress=np.asarray(r["stress"]).reshape(9);reftan=np.asarray(r["tangent"]).reshape(9)
 stress_abs=float(np.linalg.norm(direct-refstress));stress_rel=stress_abs/max(float(np.linalg.norm(refstress)),1e-300)
 state_abs=float(np.linalg.norm(state-np.asarray(r["state"])))
 tan_abs=float(np.linalg.norm(tan-reftan));tan_rel=tan_abs/max(float(np.linalg.norm(reftan)),1e-300)
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 stress_errors.append if False else None
 # collect componentwise metrics, retaining raw norm and absolute scale
 stress_errors.append({"i":i,"label":r["label"],"tet":r["tet"],"basis":r["basis_column"],"power":r["power"],"sign":r["sign"],
  "abs_l2":stress_abs,"rel_l2":stress_rel,"max_abs_component":float(np.max(np.abs(direct-refstress))),
  "reference_norm":float(np.linalg.norm(refstress)),"native_norm":float(np.linalg.norm(direct)),
  "state_abs_l2":state_abs,"tangent_abs_l2":tan_abs,"tangent_rel_l2":tan_rel,
  "direct_vs_tangent_path_max_abs":float(np.max(np.abs(direct-tpath))),
  "component_ulp_max":max(ulpdist(a,b) for a,b in zip(direct,refstress))})
 if float(np.max(np.abs(direct-tpath)))!=0:bad.append(i)
 p=f32(inp["parameters_by_material"][r["material"]]).astype(np.float64)[None,:]
 old=f32(inp["rows"][i]["accepted_state_f32"]).astype(np.float64)[None,:]
 ns=state[None,:]
 bs=sig(p,old,ns)
 if bs!=r.get("branch"):branch_mismatch.append({"i":i,"tet":r["tet"],"label":r["label"],"native":bs,"ref":r.get("branch")})
# Duplicate center exact output words.
repeat=[]
for i in range(0,len(pts),6):
 a=pts[i]["output_f32_bits"];b=pts[i+1]["output_f32_bits"]
 repeat.append({"tet":pts[i]["tet_id"],"identical_65_words":a==b,
                "different_slots":[j for j,(x,y) in enumerate(zip(a,b)) if x!=y]})
# Realized finite differences: use actual represented FP32 high+low separation.
groups={}
for i,r in enumerate(refs):
 if r["label"]=="endpoint":groups.setdefault((r["tet"],r["basis_column"],r["power"]),{})[r["sign"]]=i
center={ (r["tet"],r["basis_column"]):i for i,r in enumerate(refs) if r["label"]=="center" }
fdrows=[]
for (tet,col,power),g in sorted(groups.items()):
 ip,im=g[1],g[-1];rp,rm=refs[ip],refs[im];np_,nm=pts[ip],pts[im]
 xp=np.asarray(rp["realized_F"]).reshape(9)[col];xm=np.asarray(rm["realized_F"]).reshape(9)[col]
 sep=float(xp-xm);eps_eff=sep/2.
 sp=frombits(np_["output_f32_bits"])[8:17];sm=frombits(nm["output_f32_bits"])[8:17]
 srp=np.asarray(rp["stress"]).reshape(9);srm=np.asarray(rm["stress"]).reshape(9)
 tn=np.asarray(refs[center[(tet,col)]]["tangent"]).reshape(9)
 native_slope=(sp-sm)/sep if sep else np.full(9,np.nan)
 ref_slope=(srp-srm)/sep if sep else np.full(9,np.nan)
 nerr=float(np.linalg.norm(native_slope-tn)/max(np.linalg.norm(tn),1e-300))
 rerr=float(np.linalg.norm(ref_slope-tn)/max(np.linalg.norm(tn),1e-300))
 signal=float(np.linalg.norm(sp-sm))
 errbound=float(np.linalg.norm(sp-srp)+np.linalg.norm(sm-srm))
 same=(rp.get("branch")==rm.get("branch")==refs[center[(tet,col)]].get("branch"))
 fdrows.append({"tet":tet,"basis_column":col,"power":power,"ideal_epsilon":float(rp["epsilon"]),
  "realized_half_separation":eps_eff,"realized_separation":sep,"same_reference_branch":same,
  "native_signal_l2":signal,"sum_native_to_reference_endpoint_error_l2":errbound,
  "signal_exceeds_endpoint_error_sum":signal>errbound,
  "native_centered_fd_relative_to_strict_tangent":nerr,
  "strict_realized_endpoint_secant_relative_to_tangent":rerr,
  "native_relative_to_strict_secant":float(np.linalg.norm(native_slope-ref_slope)/max(np.linalg.norm(tn),1e-300))})
summary={"schema":"numi.cardboard.native541-critical23-adjudication.v1",
 "bindings":{"input_sha256":sha(PK/"input.json"),"stable_reference_sha256":sha(PK/"stable-fp64-reference.jsonl"),
  "native_jsonl_sha256":sha(OUT),"package_sha256":sha(PKG),"helper_sha256":sha(HELP),
  "capture_sha256":"c9a538bd6357136130fc3d971c7d2f1e61fc52bdef2a0737ba00790bd34d8ec2"},
 "native":{"rows":len(pts),"status":{"stress_valid":end["direct_stress_valid_count"],"tangent_valid":end["tangent_valid_count"],
  "projection_converged":end["tangent_projection_converged_count"]},"invalid_rows":bad,"direct_vs_tangent_internal_mismatch_rows":bad,
  "strict_reference_branch_mismatches":branch_mismatch},
 "center_repeatability":{"rows":len(repeat),"bit_identical_65word_pairs":sum(x["identical_65_words"] for x in repeat),
  "all_pairs":repeat,"interpretation":"same-run repeatability only; not an absolute arithmetic noise floor"},
 "native_vs_independent_stable_FP64":{"rows":len(stress_errors),
  "max_stress_abs_l2":max(x["abs_l2"] for x in stress_errors),"max_stress_rel_l2":max(x["rel_l2"] for x in stress_errors),
  "max_stress_component_abs":max(x["max_abs_component"] for x in stress_errors),
  "max_component_ulp_distance_vs_round_to_f32_reference":max(x["component_ulp_max"] for x in stress_errors),
  "max_projected_state_abs_l2":max(x["state_abs_l2"] for x in stress_errors),
  "max_tangent_abs_l2":max(x["tangent_abs_l2"] for x in stress_errors),
  "max_tangent_rel_l2":max(x["tangent_rel_l2"] for x in stress_errors),
  "top_stress_errors":sorted(stress_errors,key=lambda x:x["rel_l2"],reverse=True)[:12],
  "all_rows":stress_errors},
 "endpoint_derivatives":{"pairs":len(fdrows),"all_pairs":fdrows},
 "limits":["Results cover only 23 previously unresolved tet/basis pairs and two adjacent FP64-qualified epsilon powers each.",
  "Native-vs-FP64 errors combine FP32 constitutive arithmetic and local projection differences; they are measured discrepancies, not an absolute native noise floor.",
  "Duplicate center rows measure repeatability only.",
  "No claim that a tiny finite-difference signal is resolved unless it exceeds measured endpoint discrepancies and branch/epsilon behavior is stable.",
  "This is a point diagnostic, not material calibration or full coupon qualification."]}
outdir=Q/"analysis/native541-critical23-adjudication-001";outdir.mkdir(parents=True,exist_ok=True)
(outdir/"report.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"output":str(outdir/"report.json"),"sha256":sha(outdir/"report.json"),
 "rows":len(pts),"valid":end["direct_stress_valid_count"],"repeat_exact":sum(x["identical_65_words"] for x in repeat),
 "maxstressabs":summary["native_vs_independent_stable_FP64"]["max_stress_abs_l2"],
 "maxstressrel":summary["native_vs_independent_stable_FP64"]["max_stress_rel_l2"],
 "maxtangentrel":summary["native_vs_independent_stable_FP64"]["max_tangent_rel_l2"],
 "fd_pairs":len(fdrows),"signal_below_error":sum(not x["signal_exceeds_endpoint_error_sum"] for x in fdrows)},indent=2))
