#!/usr/bin/env python3
"""Read-only capture of actual 396 first-SCP matrices; optional separate 397 comparison."""
import argparse,hashlib,importlib.util,inspect,json,shlex,sys
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
OWNER=ROOT/"D/precision-phase-one-396/propose_joint_clearance_witnesses_396.py"
CMP=ROOT/"D/target-row-progress-feasibility-397/target_row_progress_feasibility_397.py"
PHASE=("--precision-phase-one-report","--precision-phase-one-rows-npz","--precision-safety-factor","--precision-cert-tolerance-mm")
PATHFLAGS={"--base-candidate","--raw-trial","--self-pairs","--area-reference-npz","--pair-owner","--area-owner","--bary-owner","--point-target-npz","--point-target-report","--face-target-npz","--face-target-report","--source-base-npz","--point-depth-rows-npz","--point-depth-rows-report","--point-depth-owner","--fixed-boundary-npz","--fixed-boundary-report","--fixed-boundary-owner"}
class Captured(Exception): pass
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for q in iter(lambda:f.read(1<<20),b""): h.update(q)
 return h.hexdigest()
def sha_array(x): return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()
def load_argv(path):
 x=json.loads(Path(path).read_text())
 if isinstance(x,dict): x=x.get("argv",x.get("command_argv",x.get("args")))
 if isinstance(x,str): x=shlex.split(x)
 if not isinstance(x,list) or not all(isinstance(q,str) for q in x): raise ValueError("argv JSON must contain string args")
 x=list(x)
 if x and Path(x[0]).name.startswith("python"): x=x[2:] if len(x)>1 and not x[1].startswith("--") else x[1:]
 return x
def prepare_argv(argv,outdir):
 out=[];removed=[];hasout=False;i=0
 while i<len(argv):
  x=argv[i]
  if x in PHASE:
   if i+1>=len(argv) or argv[i+1].startswith("--"): raise ValueError(x+" needs a value")
   removed += [x,argv[i+1]];i+=2;continue
  f=next((f for f in PHASE if x.startswith(f+"=")),None)
  if f: removed += [f,x.split("=",1)[1]];i+=1;continue
  if x=="--output":
   if i+1>=len(argv): raise ValueError("--output needs a value")
   out += [x,str(Path(outdir)/"owner-output-never-created")];hasout=True;i+=2;continue
  if x.startswith("--output="): out.append("--output="+str(Path(outdir)/"owner-output-never-created"));hasout=True;i+=1;continue
  if x=="--self-test": raise ValueError("expected normal 396 argv, not self-test")
  out.append(x);i+=1
 if not hasout: raise ValueError("normal 396 argv lacks --output")
 return out,removed
def opt(argv,key):
 for i,x in enumerate(argv):
  if x==key and i+1<len(argv):return argv[i+1]
  if x.startswith(key+"="):return x.split("=",1)[1]
 raise ValueError("missing "+key)
def target_index(query_rows,node,nclear):
 q=np.asarray(query_rows)
 if q.ndim!=1 or q.dtype.kind not in "iu": raise ValueError("query_rows must be integer vector")
 ids=np.flatnonzero(q.astype(np.int64,copy=False)==int(node))
 if len(ids)!=1: raise ValueError(f"output row {node} occurs {len(ids)} times in point-target ordering")
 if int(ids[0])>=nclear: raise ValueError("mapped point clearance row is outside C")
 return int(ids[0])
def extract_solver(args,kwargs,sig,basis):
 v=sig.bind(*args,**kwargs).arguments
 tkey="target_output" if basis else "target"; lok="lower_output" if basis else "lower"; hik="upper_output" if basis else "upper"
 keys=("A","lower_rows",tkey,lok,hik,"C","h","weights")
 if any(k not in v for k in keys):raise ValueError("actual 396 solver call signature mismatch")
 p={"A":sparse.csr_matrix(v["A"],dtype=np.float64).copy(),"b":np.asarray(v["lower_rows"],dtype=np.float64).ravel().copy(),
 "t":np.asarray(v[tkey],dtype=np.float64).ravel().copy(),"lo":np.asarray(v[lok],dtype=np.float64).ravel().copy(),
 "hi":np.asarray(v[hik],dtype=np.float64).ravel().copy(),"C":sparse.csr_matrix(v["C"],dtype=np.float64).copy(),
 "h":np.asarray(v["h"],dtype=np.float64).ravel().copy(),"w":np.asarray(v["weights"],dtype=np.float64).ravel().copy(),
 "M":sparse.csr_matrix(v["output_map"],dtype=np.float64).copy() if basis else None,"basis":basis}
 n=p["A"].shape[1]
 if p["A"].shape[0]!=len(p["b"]) or p["t"].shape!=(n,) or p["lo"].shape!=(n,) or p["hi"].shape!=(n,) or p["C"].shape[1]!=n or p["C"].shape[0]!=len(p["h"]) or p["h"].shape!=p["w"].shape or np.any(p["lo"]>p["hi"]):raise ValueError("captured 396 matrices/vectors misaligned")
 if p["M"] is not None and p["M"].shape[0]!=n:raise ValueError("basis output dimension mismatch")
 if not all(np.isfinite(p[k]).all() for k in ("b","t","lo","hi","h","w")) or not np.isfinite(p["A"].data).all() or not np.isfinite(p["C"].data).all():raise ValueError("nonfinite captured LP input")
 return p
def small_identity(row):
 out={}
 for k,v in row.items():
  if k in ("nodes","coefficients"): continue
  if isinstance(v,np.generic): v=v.item()
  if isinstance(v,np.ndarray) and v.ndim<=2 and v.size<=32: v=v.tolist()
  if v is None or isinstance(v,(str,int,float,bool)): out[k]=v
  elif isinstance(v,(list,tuple)) and all(z is None or isinstance(z,(str,int,float,bool)) for z in v): out[k]=list(v)
 return out
def capture(owner_path,argv,outdir,argv_json,node,removed):
 outdir=Path(outdir)
 if outdir.exists():raise FileExistsError("capture directory exists: "+str(outdir))
 owner_dir=str(Path(owner_path).resolve().parent)
 if owner_dir not in sys.path:sys.path.insert(0,owner_dir)
 spec=importlib.util.spec_from_file_location("owner396_capture",owner_path)
 if spec is None or spec.loader is None:raise RuntimeError("cannot import 396 owner")
 owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
 with np.load(opt(argv,"--point-target-npz"),allow_pickle=False) as z:qrows=np.asarray(z["query_rows"]).copy()
 state={"p":None,"solver":None,"pairmeta":[],"npair":0,"narea":0,"area_faces":[]}
 oldmat=owner._owner_matrix
 def matrix_hook(pair_rows,area_rows,area_rhs,support):
  A,b=oldmat(pair_rows,area_rows,area_rhs,support);state["pairmeta"]=[small_identity(x) for x in pair_rows];state["npair"]=len(pair_rows);state["narea"]=int(area_rows.shape[0]);return A,b
 owner._owner_matrix=matrix_hook
 oldload=owner.load_module
 def load_hook(name,path):
  m=oldload(name,path)
  if "area_linearization" in name:
   old=m.area_linearization
   def area_hook(points,faces,ids,*a,**kw):
    state["area_faces"]=np.asarray(ids,dtype=np.int64).ravel().tolist()
    return old(points,faces,ids,*a,**kw)
   m.area_linearization=area_hook
  return m
 owner.load_module=load_hook
 for name,basis in (("solve_sparse_l1_clearance",False),("solve_sparse_l1_clearance_in_basis",True)):
  orig=getattr(owner,name);sig=inspect.signature(orig)
  def hook(*a,__name=name,__basis=basis,__sig=sig,**kw):
   state["p"]=extract_solver(a,kw,__sig,__basis);state["solver"]=__name;raise Captured()
  setattr(owner,name,hook)
 oldargv=sys.argv
 try:
  sys.argv=[str(owner_path),*argv]
  try:owner.main()
  except Captured:pass
 finally:sys.argv=oldargv
 if state["p"] is None:raise RuntimeError("396 normal assembly never reached solver")
 p=state["p"];ri=target_index(qrows,node,p["C"].shape[0])
 if not np.isfinite(p["h"][ri]) or p["h"][ri]<=0:raise ValueError("target h is not positive")
 outdir.mkdir(parents=True,exist_ok=False)
 sparse.save_npz(outdir/"hard_A.npz",p["A"],compressed=True);sparse.save_npz(outdir/"clearance_C.npz",p["C"],compressed=True)
 if p["M"] is not None:sparse.save_npz(outdir/"output_map_M.npz",p["M"],compressed=True)
 np.savez_compressed(outdir/"vectors.npz",b=p["b"],raw_target=p["t"],lower=p["lo"],upper=p["hi"],h=p["h"],weights=p["w"],target_row=np.int64(ri),target_output_node=np.int64(node),point_query_rows=qrows)
 names=["hard_A.npz","clearance_C.npz","vectors.npz"]+(["output_map_M.npz"] if p["M"] is not None else [])
 files={x:sha_file(outdir/x) for x in names}; inputs={}
 for i,x in enumerate(argv):
  f,eq,val=x.partition("=")
  if f in PATHFLAGS:
   val=val if eq else (argv[i+1] if i+1<len(argv) else "")
   if Path(val).is_file():inputs[str(Path(val))]=sha_file(val)
 npair=state["npair"];narea=state["narea"]
 report={"schema":"numi.human.396-normal-assembly-capture.v1","classification":"read-only matrix capture before solver; no candidate emitted","solver_invoked":False,"solver_intercepted":state["solver"],"physical_gates":"unchanged/not run","first_scp_linearization_only":True,
 "target_output_node":int(node),"clearance_row_index":ri,"row_mapping":"query_rows order; 396 point C rows precede face C rows","target_h":float(p["h"][ri]),
 "uses_immediate_parent_basis":p["basis"],"output_dimension":p["A"].shape[1],"control_dimension":p["M"].shape[1] if p["M"] is not None else p["A"].shape[1],
 "A_shape":list(p["A"].shape),"A_nnz":p["A"].nnz,"C_shape":list(p["C"].shape),"C_nnz":p["C"].nnz,
 "hard_A_ranges":{"pair_and_point_barriers":[0,npair],"fixed_037_area":[npair,npair+narea],"edge_gradient":[npair+narea,p["A"].shape[0]]},
 "pair_and_point_barrier_row_metadata":state["pairmeta"],"area_face_rows":state["area_faces"],"matrix_file_sha256":files,
 "vector_sha256":{k:sha_array(p[v]) for k,v in (("b","b"),("target","t"),("lower","lo"),("upper","hi"),("h","h"),("weights","w"))},
 "owner_396_path":str(Path(owner_path).resolve()),"owner_396_sha256":sha_file(owner_path),"helper_397_path":str(CMP),"helper_397_sha256":sha_file(CMP),
 "argv_json_path":str(Path(argv_json).resolve()),"argv_json_sha256":sha_file(argv_json),"normal_argv":list(argv),"removed_precision_phase_one_flags":list(removed),"normal_input_file_sha256":inputs}
 rp=outdir/"assembly-capture.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 return {"capture_dir":str(outdir),"report_path":str(rp),"report_sha256":sha_file(rp)}
def dual_summary(dual,counts,names,report,limit=25):
 if dual is None:return {"status":"unavailable"}
 v=np.asarray(dual,dtype=np.float64);offsets={};i=0
 for name,n in zip(names,counts):offsets[name]=(i,i+int(n));i+=int(n)
 hs,he=offsets.get("hard_A",(0,0))
 if i!=len(v):return {"status":"row-layout-mismatch","dual_rows":len(v),"expected":i}
 ranges=report["hard_A_ranges"];meta=report["pair_and_point_barrier_row_metadata"];areas=report["area_face_rows"];top=[]
 for local in np.argsort(np.abs(v[hs:he]))[::-1]:
  mag=abs(float(v[hs+local]))
  if mag<=1e-10:break
  r=int(local)
  if ranges["pair_and_point_barriers"][0]<=r<ranges["pair_and_point_barriers"][1]:
   cat="pair_or_point_barrier";j=r-ranges["pair_and_point_barriers"][0];ident=meta[j] if j<len(meta) else None
  elif ranges["fixed_037_area"][0]<=r<ranges["fixed_037_area"][1]:
   cat="fixed_037_area";j=r-ranges["fixed_037_area"][0];ident={"area_face_row":areas[j]} if j<len(areas) else None
  else:cat="edge_gradient";ident={"local_row":r-ranges["edge_gradient"][0]}
  top.append({"A_row":r,"category":cat,"identity":ident,"dual_magnitude":mag})
  if len(top)>=limit:break
 return {"status":"available","top_nonzero_hard_rows":top,"all_hard_duals_zero":not bool(top)}
def compare_capture(capture_dir,time_limit_s=30.,lp_method="highs"):
 d=Path(capture_dir);rp=d/"assembly-capture.json";r=json.loads(rp.read_text())
 if r.get("schema")!="numi.human.396-normal-assembly-capture.v1" or r.get("solver_invoked") is not False:raise ValueError("not a valid pre-solver capture")
 for name,digest in r["matrix_file_sha256"].items():
  if sha_file(d/name)!=digest:raise ValueError("matrix SHA mismatch: "+name)
 spec=importlib.util.spec_from_file_location("progress397",CMP)
 if spec is None or spec.loader is None:raise RuntimeError("cannot import 397 comparator")
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 A=sparse.load_npz(d/"hard_A.npz");C=sparse.load_npz(d/"clearance_C.npz")
 with np.load(d/"vectors.npz",allow_pickle=False) as z:b,lo,hi,h=(np.asarray(z[k]).copy() for k in ("b","lower","upper","h"));row=int(z["target_row"])
 M=sparse.load_npz(d/"output_map_M.npz") if r["uses_immediate_parent_basis"] else None
 ans=mod.compare_target_progress_feasibility(A,b,lo,hi,M,C,h,row,time_limit_s=time_limit_s,lp_method=lp_method)
 vectors={}
 for label,key in (("without","without_global_no_regression"),("with","with_global_no_regression")):
  part=ans["results"][key];du=part.pop("inequality_dual_marginals",None);res=part.pop("inequality_primal_residuals",None);delta=part.pop("primal_delta_output",None);q=part.pop("primal_control_coordinates",None)
  if du is not None:vectors[label+"_duals"]=np.asarray(du,dtype=np.float64)
  if res is not None:vectors[label+"_inequality_residuals"]=np.asarray(res,dtype=np.float64)
  if delta is not None:vectors[label+"_delta_output"]=np.asarray(delta,dtype=np.float64)
  if q is not None:vectors[label+"_controls"]=np.asarray(q,dtype=np.float64)
  part["top_hard_dual_witnesses"]=dual_summary(du,part["constraint_row_counts"],part["constraint_blocks"],r)
 if (d/"comparison.json").exists() or (d/"comparison-vectors.npz").exists():raise FileExistsError("comparison already written")
 np.savez_compressed(d/"comparison-vectors.npz",**vectors)
 ans["comparison_vector_sha256"]=sha_file(d/"comparison-vectors.npz")
 ans["source_pins"]={"capture_report_sha256":sha_file(rp),"owner_396_sha256":r["owner_396_sha256"],"helper_397_sha256":r["helper_397_sha256"],"matrix_file_sha256":r["matrix_file_sha256"],"target_output_node":r["target_output_node"]}
 ans["diagnostic_only"]=True;ans["physical_geometry_gates"]="unchanged; linear feasibility only"
 (d/"comparison.json").write_text(json.dumps(ans,indent=2,sort_keys=True,allow_nan=False)+"\n")
 return ans
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--argv-json",type=Path);ap.add_argument("--capture-dir",type=Path);ap.add_argument("--owner",type=Path,default=OWNER);ap.add_argument("--target-output-node",type=int,default=18706);ap.add_argument("--compare-capture",type=Path);ap.add_argument("--time-limit-s",type=float,default=30.);ap.add_argument("--lp-method",choices=("highs","highs-ipm","highs-ds"),default="highs")
 a,rest=ap.parse_known_args()
 if a.compare_capture is not None:
  if a.argv_json or a.capture_dir or rest:ap.error("--compare-capture cannot combine with capture argv")
  print(json.dumps(compare_capture(a.compare_capture,a.time_limit_s,a.lp_method),indent=2,sort_keys=True));return
 if a.argv_json is None or a.capture_dir is None or rest:ap.error("--argv-json and --capture-dir required")
 argv,removed=prepare_argv(load_argv(a.argv_json),a.capture_dir)
 print(json.dumps(capture(a.owner,argv,a.capture_dir,a.argv_json,a.target_output_node,removed),indent=2,sort_keys=True))
if __name__=="__main__":main()
