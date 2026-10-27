"""Bounded graph-distance damping around exact self-intersection witnesses."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, subprocess, sys, time
from collections import deque, Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
D=HERE.parent
ROOT=D.parents[1]
BASE_STATE= D/"connected-topology-580/run-010-from689/root-combined-001/final/state.json"
PROPOSAL_DIR=D/"component-sat-selector-638/outward-boneface-filter-640/root-bank43-component-001"
ARITH=ROOT/"D/publication-177/sources/run_candidate_checks545.py"
AUDIT130=ROOT/"audit_connected_outer_geometry_130.py"
STAGE_PATH=D/"continue_surface_round546.py"
PROFILES={
 "restore_witness_ramp":{0:0.0,1:0.25,2:0.55,3:0.80,4:1.0},
 "light_witness_ramp":{0:0.10,1:0.35,2:0.65,3:0.85,4:0.98,5:1.0},
 "partial_witness_ramp":{0:0.20,1:0.45,2:0.75,3:1.0},
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb")as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def same(a,b):
 a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
 return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise ImportError(path)
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def write(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")

def node_graph(faces,n):
 adj=[set() for _ in range(n)]
 for tri in np.asarray(faces):
  a,b,c=map(int,tri)
  adj[a].update((b,c));adj[b].update((a,c));adj[c].update((a,b))
 return adj
def graph_distances(adj,seeds):
 dist={int(s):0 for s in seeds};q=deque(sorted(dist))
 while q:
  u=q.popleft()
  for v in sorted(adj[u]):
   if v not in dist:dist[v]=dist[u]+1;q.append(v)
 return dist

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
 out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 state=json.loads(BASE_STATE.read_text()); receipt=json.loads((PROPOSAL_DIR/"receipt.json").read_text())
 proposal=PROPOSAL_DIR/"candidate.npz"
 pairfile=PROPOSAL_DIR/"full-check/audit-130/surface-self-pairs.npy"
 audit130summary=PROPOSAL_DIR/"full-check/audit-130/summary.json"
 fullcheck=json.loads((PROPOSAL_DIR/"full-check/report.json").read_text())
 if sha(state["candidate"])!=state["candidate_sha256"] or sha(BASE_STATE)!=state.get("state_sha256",sha(BASE_STATE)):
  raise ValueError("baseline candidate/state input hash mismatch")
 if sha(proposal)!=receipt["candidate_sha256"] or fullcheck.get("candidate_sha256")!=receipt["candidate_sha256"]:
  raise ValueError("proposal receipt/candidate mismatch")
 if sha(pairfile)=="" or json.loads(audit130summary.read_text()).get("candidate_sha256")!=receipt["candidate_sha256"]:
  raise ValueError("self witness file/report mismatch")
 with np.load(state["candidate"],allow_pickle=False)as z:base={k:np.array(z[k],copy=True)for k in z.files}
 with np.load(proposal,allow_pickle=False)as z:src={k:np.array(z[k],copy=True)for k in z.files}
 if set(base)!=set(src):raise ValueError("baseline/proposal archive key mismatch")
 for k in base:
  if k!="points_world_m" and not same(base[k],src[k]):raise ValueError("proposal changed nonpoint payload: "+k)
 bf=base["source_O_faces"];bx=base["points_world_m"];sx=src["points_world_m"]
 if bx.dtype!=np.float32 or sx.dtype!=np.float32 or bx.shape!=sx.shape or not same(bf,src["source_O_faces"]):
  raise ValueError("point or topology domain mismatch")
 changed=np.flatnonzero(np.any(sx.view(np.uint32)!=bx.view(np.uint32),axis=1))
 if not np.array_equal(changed,np.asarray(receipt["retained_node_rows"],dtype=np.int64)):
  raise ValueError("changed-node rows differ from selector receipt")
 anchors=np.asarray(base["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
 if len(anchors)!=237 or not same(bx[anchors],sx[anchors]):raise ValueError("typed anchor inventory/motion mismatch")
 pairs=np.load(pairfile,allow_pickle=False)
 if pairs.shape!=(15,2) or not np.array_equal(np.sort(pairs,axis=1),pairs):
  # Pair ordering is the exact owner's canonical face-row order.
  raise ValueError("unexpected exact self witness shape/order")
 witness_faces=np.unique(pairs)
 seed_nodes=np.unique(bf[witness_faces].ravel())
 adj=node_graph(bf,len(bx));dist=graph_distances(adj,seed_nodes)
 if any(int(row) not in dist for row in changed):
  raise ValueError("some changed nodes have no face-graph distance to self witnesses")
 py=loadmod("continue_surface_round546_component_damping",STAGE_PATH)
 input_paths=[BASE_STATE,Path(state["candidate"]),Path(state["surface_report"]),Path(state["point_summary"]),
    PROPOSAL_DIR/"receipt.json",proposal,pairfile,audit130summary,PROPOSAL_DIR/"full-check/report.json",
    ARITH,AUDIT130,STAGE_PATH,Path(__file__).resolve()]
 input_pins={str(p.resolve()):sha(p)for p in input_paths}
 launch={"schema":"numi.self_witness_local_damping.launch.v1",
   "baseline_state":str(BASE_STATE),"baseline_state_sha256":sha(BASE_STATE),
   "baseline_candidate_sha256":state["candidate_sha256"],"proposal_path":str(proposal),
   "proposal_sha256":sha(proposal),"proposal_self_pair_path":str(pairfile),"proposal_self_pair_sha256":sha(pairfile),
   "self_pair_count":len(pairs),"witness_face_rows":witness_faces.tolist(),"witness_node_rows":seed_nodes.tolist(),
   "changed_node_count":len(changed),"graph_distance_counts":{str(k):int(v)for k,v in sorted(Counter(dist[int(i)]for i in changed).items())},
   "profiles":PROFILES,"source_pins":input_pins,"native_admission":False,"accepted_physical_steps":0}
 write(out/"launch.json",launch)
 rows=[]
 for name,table in PROFILES.items():
  vd=out/name;vd.mkdir()
  weights=np.ones(len(bx),dtype=np.float64)
  for row in changed:
   di=int(dist[int(row)])
   weights[int(row)]=table.get(di,1.0)
  xnew=(bx.astype(np.float64)+weights[:,None]*(sx.astype(np.float64)-bx.astype(np.float64))).astype(np.float32)
  if not same(xnew[anchors],bx[anchors]):raise ValueError("damped variant moved anchors")
  arrays={k:np.array(v,copy=True)for k,v in base.items()};arrays["points_world_m"]=xnew
  cp=vd/"candidate.npz";np.savez_compressed(cp,**arrays);csha=sha(cp)
  changed_new=np.flatnonzero(np.any(xnew.view(np.uint32)!=bx.view(np.uint32),axis=1))
  variant={"name":name,"candidate_path":str(cp),"candidate_sha256":csha,
    "changed_rows":changed_new.tolist(),"changed_row_count":len(changed_new),
    "weights_by_graph_distance":{str(k):float(v)for k,v in table.items()},
    "max_step_mm":float(np.linalg.norm((xnew.astype(np.float64)-bx.astype(np.float64))*1000,axis=1).max(initial=0.0)),
    "baseline_nonpoint_payload_bitwise":all(same(arrays[k],base[k])for k in base if k!="points_world_m"),
    "anchors_bitwise_unchanged":same(xnew[anchors],bx[anchors]),
    "native_admission":False,"accepted_physical_steps":0}
  vsource=(Path(__file__).resolve(),BASE_STATE,Path(state["candidate"]),Path(state["surface_report"]),Path(state["point_summary"]),
    proposal,PROPOSAL_DIR/"receipt.json",ARITH,STAGE_PATH)
  argv=[sys.executable,str(ARITH),"--candidate",str(cp),"--output-dir",str(vd/"arithmetic"),
        "--arithmetic-only","--trust-mm","4","--baseline-state",str(BASE_STATE)]
  try:
   py.run_stage(argv,vd/"arithmetic-stage",slot=False,source_paths=vsource,timeout=60)
   ar=json.loads((vd/"arithmetic/report.json").read_text())
   if ar.get("status")!="ARITHMETIC_PASS" or ar.get("candidate_sha256")!=csha:
    variant.update(arithmetic_status=ar.get("status"),arithmetic_error=ar.get("error"),geometry_status="NOT_RUN")
   else:
    variant["arithmetic_status"]="ARITHMETIC_PASS"
    a130=vd/"audit-130"
    argv130=[sys.executable,str(AUDIT130),str(cp),str(a130)]
    py.run_stage(argv130,vd/"audit-130-stage",slot=True,
      source_paths=(Path(__file__).resolve(),AUDIT130,cp,BASE_STATE,Path(state["candidate"]),STAGE_PATH),timeout=120)
    s130=json.loads((a130/"summary.json").read_text())
    selfpairs=np.load(a130/"surface-self-pairs.npy",allow_pickle=False)
    if s130.get("candidate_sha256")!=csha or s130.get("complete_surface_audit") is not True or len(selfpairs)!=s130.get("exact_PLC_surface_pairs"):
      raise ValueError("candidate-bound complete130 receipt mismatch")
    variant.update(geometry_status=s130.get("status"),self_pair_count=int(len(selfpairs)),
      inside_node_bank_pairs=s130.get("inside_node_bank_pairs"),exact_130_summary_sha256=sha(a130/"summary.json"),
      exact_self_pairs_path=str(a130/"surface-self-pairs.npy"),exact_self_pairs_sha256=sha(a130/"surface-self-pairs.npy"),
      eligible_for_bank43_screen=len(selfpairs)==0)
  except Exception as exc:
   variant.update(stage_error=f"{type(exc).__name__}: {exc}",geometry_status="FAILED_OR_NOT_RUN")
  rows.append(variant);write(out/"progress.json",{"variants":rows,"native_admission":False,"accepted_physical_steps":0})
 for p,h in input_pins.items():
  if sha(p)!=h:raise RuntimeError("source/input changed: "+p)
 result={"schema":"numi.self_witness_local_damping.result.v1","baseline_state":str(BASE_STATE),
  "baseline_candidate_sha256":state["candidate_sha256"],"proposal_candidate_sha256":sha(proposal),
  "proposal_fullcheck_candidate_sha256":fullcheck.get("candidate_sha256"),"variants":rows,
  "source_pins_unchanged":True,"comparison_to_bank43_crossings_not_run":True,
  "native_admission":False,"accepted_physical_steps":0}
 write(out/"report.json",result)
 print(json.dumps({"status":"DONE","variants":[{k:v for k,v in z.items()if k in ["name","arithmetic_status","geometry_status","self_pair_count","inside_node_bank_pairs","eligible_for_bank43_screen","stage_error"]}for z in rows]},sort_keys=True),flush=True)

if __name__=="__main__":main()

