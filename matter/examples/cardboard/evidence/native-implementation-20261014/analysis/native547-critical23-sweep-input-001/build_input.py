import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PK=Q/"analysis/native547-critical23-sweep-input-001"; SRC=Q/"snapshots/source-cohesive-quadrature-area-205"
CIN=Q/"analysis/native541-center-input-001/input.json"; FINE=Q/"analysis/native531-stable-green-strain-reference-017-001/extended23-v2/report.json"
STABLE=Q/"analysis/native531-stable-green-strain-reference-017-001/reference/cardboard_bilinear_fp64_reference.py"
sys.path.insert(0,str(SRC/"matter/tools"))
import cardboard_compensated_fp64_reference as strict
spec=importlib.util.spec_from_file_location("stable_green_reference",STABLE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);engine=strict.Reference(base)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x):return np.asarray(x,dtype=np.float32)
def bits(x):return "0x%08x"%int(np.asarray(x,dtype=np.float32).view(np.uint32))
def branch(p,o,z):
 b=base.paper_branch(base,p,o,z);dl=b["plastic_multiplier_increment"];y=b["yield_function_residual_value"]
 return [float(np.sign(dl[0])),float(np.sign(y[0])),bool((np.sqrt(dl*dl+y*y+p[:,26]**2)+dl+y)[0]>1e-14)]
source=json.loads(CIN.read_text());fine=json.loads(FINE.read_text())
byid={(int(x["environment"]),int(x["tet_id"])):x for x in source["rows"]}
params=[f32(v).astype(np.float64) for v in source["parameters_by_material"]]
rows=[];refs=[];pairs=[]
for item in fine["results"]:
 tet=int(item["tet"]);col=int(item["basis_column"]);center=byid[(0,tet)]
 mi=int(center["material_index"]);w=f32(center["packed_input_f32"]);aw=f32(center["accepted_state_f32"])
 old=aw.astype(np.float64)[None,:];p=params[mi][None,:];dt=float(w[45])
 H=np.zeros((3,3),dtype=np.float64);H.flat[col]=1.0
 qualifies=list(map(int,item["qualifying_powers"]));bp=int(item["best_power"])
 adjacent=bp+1 if bp+1 in qualifies else bp-1 if bp-1 in qualifies else None
 if adjacent is None:raise RuntimeError(f"no adjacent p for {tet}/{col}")
 powers=list(range(10,41));pairs.append({"tet_id":tet,"material_index":mi,"basis_column":col,
  "powers":powers,"best_power":bp,"best_fp64_relative_error":float(item["best_relative_error"])})
 for label,power,sign in [("center",None,0),("center_repeat",None,0)]+[("endpoint",q,s) for q in powers for s in (1,-1)]:
  x=w.copy();x[27:36]=f32(H.reshape(9));x[36:45]=f32((H/dt).reshape(9))
  cf=(w[:9].astype(np.float64)+w[9:18].astype(np.float64)).reshape(3,3);target=cf.copy();eps=None
  if label=="endpoint":
   eps=2.0**(-int(power));target.flat[col]+=sign*eps
   hi=f32(w[:9]);lo=f32(w[9:18]);hi[col]=np.float32(target.flat[col])
   lo[col]=np.float32((cf.flat[col]-float(hi[col]))+sign*eps)
   x[:9]=hi;x[9:18]=lo
   rate=w[18:27].astype(np.float64).copy();rate.flat[col]+=sign*eps/dt;x[18:27]=f32(rate)
  rf=(x[:9].astype(np.float64)+x[9:18].astype(np.float64)).reshape(3,3);rd=x[18:27].astype(np.float64)
  fp=rf[None,:,:];z,it,rn,cn,ok,why,_=engine.project_with_metrics(p,fp,old)
  ref={"index":len(rows),"label":label,"env":0,"tet":tet,"material":mi,"basis_column":col,
   "power":power,"sign":sign,"epsilon":eps,"input_f32_hex":[bits(a) for a in x],
   "accepted_f32_hex":[bits(a) for a in aw],"center_F":cf.tolist(),"target_F":target.tolist(),
   "realized_F":rf.tolist(),"realized_delta_F":(rf-cf).tolist(),"realized_minus_ideal_F":(rf-target).tolist(),
   "realized_Fdot":rd.tolist(),"realized_delta_Fdot":(rd-w[18:27].astype(np.float64)).tolist(),
   "strict_valid":bool(ok[0]),"reason":str(why[0]),"scaled_residual":float(rn[0]),
   "dynamic_correction":float(cn[0]),"iterations":int(it[0])}
  if ok[0]:
   stress=base.stress(p,fp,z)[0];tangent=strict.tangent(base,p,fp,old,z,H[None,:,:])[0]
   ref.update(branch=branch(p,old,z),state=z[0].tolist(),stress=stress.tolist(),tangent=tangent.tolist(),
    stress_norm=float(np.linalg.norm(stress)),tangent_norm=float(np.linalg.norm(tangent)))
  rows.append({"environment":0,"tet_id":tet,"material_index":mi,"packed_input_f32":[float(v) for v in x],
   "accepted_state_f32":[float(v) for v in aw],"label":label,"basis_column":col,"epsilon_power":power,"sign":sign})
  refs.append(ref)
PK.mkdir(parents=True,exist_ok=True)
out={"schema":"numi.cardboard.source-bound-native-batch.v1","source_capture_sha256":source["source_capture_sha256"],
 "compiled_package_sha256":source["compiled_package_sha256"],
 "parameters_by_material":[[float(v) for v in f32(a)] for a in source["parameters_by_material"]],"rows":rows}
(PK/"input.json").write_text(json.dumps(out,separators=(",",":"),allow_nan=False)+"\n")
with (PK/"stable-fp64-reference.jsonl").open("w") as f:
 for r in refs:f.write(json.dumps(r,separators=(",",":"),allow_nan=False)+"\n")
(PK/"pair-selection.json").write_text(json.dumps({"schema":"numi.cardboard.native547-critical23-power-sweep.v1",
 "center_input_sha256":sha(CIN),"fine_report_sha256":sha(FINE),"pairs":pairs,
 "order":"center, exact duplicate center, then plus/minus endpoints for every binary power10 through40; broad sweep is not preselected by native outcomes"},indent=2)+"\n")
print(json.dumps({"rows":len(rows),"pairs":len(pairs),"strict_valid":sum(r["strict_valid"] for r in refs),
 "invalid":[{"tet":r["tet"],"basis":r["basis_column"],"label":r["label"],"power":r["power"],
 "reason":r["reason"],"residual":r["scaled_residual"],"correction":r["dynamic_correction"]} for r in refs if not r["strict_valid"]],
 "input_sha256":sha(PK/"input.json"),"reference_sha256":sha(PK/"stable-fp64-reference.jsonl"),
 "selection_sha256":sha(PK/"pair-selection.json")},indent=2))
