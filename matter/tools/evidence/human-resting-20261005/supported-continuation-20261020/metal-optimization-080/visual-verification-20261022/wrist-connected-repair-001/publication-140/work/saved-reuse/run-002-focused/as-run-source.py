#!/usr/bin/env python3
"""Reuse an existing coordinate proposal inside exact-witness anatomical patches."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
FOCUS={(109,114),(115,118),(65,99),(43,16),(124,123),(59,95)}
G=ROOT/"D/graded-local-step-423/graded_local_step_423.py"
G_SHA="c838c0b82931532d"  # exact source hash is captured and checked around invocation
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def focus_scope(faces,report):
 if report.get("complete") is not True or report.get("bank_count")!=185 or report.get("expected_bank_count")!=185:raise ValueError("incomplete typed bank report")
 banks=report["banks"]
 if len(banks)!=185 or len({(int(b["body"]),int(b["stable_bone_id"])) for b in banks})!=185:raise ValueError("duplicate typed bank")
 seeds=set();seen=set()
 for b in banks:
  key=(int(b["body"]),int(b["stable_bone_id"]))
  if key not in FOCUS:continue
  seen.add(key);r=b.get("result") or {"count":0,"triangle_pairs":[]}
  if r["count"]!=len(r["triangle_pairs"]):raise ValueError("typed pair table/count mismatch")
  seeds.update(int(p[0]) for p in r["triangle_pairs"])
 if seen!=FOCUS or not seeds or min(seeds)<0 or max(seeds)>=len(faces):raise ValueError("invalid focused face domain")
 e={}
 for i,t in enumerate(faces):
  for u,v in zip(t,np.roll(t,-1)):e.setdefault(tuple(sorted((int(u),int(v)))),[]).append(i)
 scope=set(seeds)
 for rows in e.values():
  if len(rows)>2:raise ValueError("nonmanifold scope")
  if len(rows)==2 and any(i in seeds for i in rows):scope.update(rows)
 return np.array(sorted(seeds),dtype=np.int64),np.array(sorted(scope),dtype=np.int64)
def main():
 ap=argparse.ArgumentParser()
 for name in ["parent","surface-report","proposal","proposal-origin","self-witnesses","output-dir"]:ap.add_argument("--"+name,required=True)
 ap.add_argument("--alpha",type=float,default=.125)
 a=ap.parse_args();out=Path(a.output_dir)
 if out.exists():raise FileExistsError(out)
 if not 0<a.alpha<=1:raise ValueError("invalid alpha")
 out.mkdir(parents=True);start=time.monotonic()
 paths=[Path(a.parent),Path(a.surface_report),Path(a.proposal),Path(a.proposal_origin),Path(a.self_witnesses),G,Path(__file__)]
 pins={str(p):sha(p) for p in paths}
 b=load(a.parent);o=load(a.proposal_origin);p=load(a.proposal);j=json.loads(Path(a.surface_report).read_text())
 if j["candidate_sha256"]!=sha(a.parent):raise ValueError("stale surface report")
 f=b["source_O_faces"];x=b["points_world_m"];anchors=b["typed_coarse_anchor_output_rows_018"]
 if not (x.dtype==o["points_world_m"].dtype==p["points_world_m"].dtype==np.float32) or x.shape!=p["points_world_m"].shape or x.shape!=o["points_world_m"].shape:raise ValueError("coordinate domain mismatch")
 if not np.array_equal(x.view(np.uint32),o["points_world_m"].view(np.uint32)):raise ValueError("saved proposal origin points differ from current parent")
 seeds,scope=focus_scope(f,j);nodes=np.unique(f[scope])
 g=module("graded423_focused_reuse",G);edges=g._edges_from_faces(f)
 d=p["points_world_m"].astype(np.float64)-x.astype(np.float64)
 el=np.linalg.norm(x[edges[:,0]].astype(np.float64)-x[edges[:,1]].astype(np.float64),axis=1)
 amp=np.maximum(np.linalg.norm(d[edges[:,0]],axis=1),np.linalg.norm(d[edges[:,1]],axis=1))
 budget=np.ones(len(edges));moving=amp>0;budget[moving]=np.minimum(1.,.1*el[moving]/amp[moving])
 ctx=out/"context.npz";np.savez_compressed(ctx,faces=f,edges=edges,edge_scale_budget=budget,points_baseline_m=x,proposal_displacement_m=d,anchors=anchors)
 os.environ.update(NUMI_GRADED_LOCAL_STEP_CONTEXT_423=str(ctx),NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423=sha(ctx),NUMI_GRADED_LOCAL_STEP_KAPPA_423=".1")
 scales=np.zeros(len(x));scales[nodes]=a.alpha;scales[anchors]=0.
 pairs=np.load(a.self_witnesses,allow_pickle=False)
 # These retained exact failure faces drive the existing reducer; min-plus
 # grading keeps the newly fixed outside-region nodes from creating seams.
 scales,graded=g.reduce_witness_node_scales(f,scales,np.empty(0,dtype=np.int64),pairs,anchors=anchors)
 y=(x.astype(np.float64)+scales[:,None]*d).astype(np.float32);y[anchors]=x[anchors]
 outside=np.ones(len(x),bool);outside[nodes]=False
 if not np.array_equal(y[outside].view(np.uint32),x[outside].view(np.uint32)):raise ValueError("outside-region positions changed")
 c={**b,"points_world_m":y};cp=out/"candidate.npz";np.savez_compressed(cp,**c);np.save(out/"node-step-scales.npy",scales)
 if any(sha(k)!=v for k,v in pins.items()):raise ValueError("source/input changed")
 result={"candidate":str(cp),"candidate_sha256":sha(cp),"input_pins":pins,"alpha_upper":a.alpha,"focus_witness_face_count":len(seeds),"scope_face_count":len(scope),"scope_node_count":len(nodes),"changed_point_count":int(np.any(y!=x,axis=1).sum()),"outside_scope_bitwise_unchanged":True,"anchors_unchanged":True,"noncoordinate_arrays_unchanged":True,"graded_receipt":graded,"seconds":time.monotonic()-start,"native_admission":False,"accepted_physical_steps":0}
 (out/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({k:v for k,v in result.items() if k not in ["input_pins","graded_receipt"]}))
if __name__=="__main__":main()
