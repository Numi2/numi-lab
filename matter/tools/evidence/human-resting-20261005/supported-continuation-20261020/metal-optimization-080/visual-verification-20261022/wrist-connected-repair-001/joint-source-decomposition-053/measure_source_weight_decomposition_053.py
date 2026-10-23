#!/usr/bin/env python3
"""Exact affine/weight edge decomposition of current294 native source poses.
Initialization diagnostics only: never changes skin, FEM or physical state.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;E=HERE.parent
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.skin_source_payload_preflight import decode_payload
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
lbs=load_module("native_source_lbs",E/"audit_bilateral_skin_reweighted_candidate_016.py")
stretch=load_module("joint_stretch",E/"current294-joint-source-check-052/measure_registered_joint_stretch.py")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def affines(d,snapshot):
 state={int(x["body_index"]):x for x in snapshot["bodies"]}
 A=[];t=[]
 for binding,body in zip(d["bindings_f"],d["bindings_u"][:,0]):
  s=state[int(body)]["current"];R=lbs.mat(s["orientation_world_xyzw"])
  A.append(R@lbs.mat(binding[4:8])*float(binding[8]));t.append(R@binding[1:4].astype(float)+s["position_world_m"])
 return np.asarray(A),np.asarray(t)
def replay(V,W,A,t):
 P=np.zeros_like(V,dtype=float)
 for j in range(len(A)):P+=W[:,j,None]*(V@A[j].T+t[j])
 return P
def edge_decomposition(x0,x1,w0,w1,A,t):
 midpoint=(x0+x1)*.5;dx=x1-x0;dw=w1-w0;wm=(w0+w1)*.5
 geometry=np.einsum("b,bij,j->i",wm,A,dx)
 terms=dw[:,None]*(np.einsum("bij,j->bi",A,midpoint)+t)
 return geometry,terms.sum(axis=0),terms
def self_test():
 rng=np.random.default_rng(782)
 for _ in range(20):
  V=rng.normal(size=(2,3));W=rng.random((2,7));W/=W.sum(axis=1)[:,None];A=rng.normal(size=(7,3,3));t=rng.normal(size=(7,3))
  p=replay(V,W,A,t);g,w,terms=edge_decomposition(V[0],V[1],W[0],W[1],A,t)
  if not np.allclose(g+w,p[1]-p[0],rtol=1e-12,atol=1e-12):raise AssertionError("edge identity")
  if not np.allclose(terms.sum(axis=0),w,rtol=0,atol=0):raise AssertionError("per-binding sum")
 print("affine_weight_edge_decomposition=PASS",flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true");a=p.parse_args()
 if a.self_test:self_test();return
 inv=json.loads((HERE/"reference/invocation.json").read_text());argv=inv["argv"];payload=Path(argv[argv.index("--skin-payload")+1])
 if sha(payload)!="d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04":raise ValueError("wrong source")
 d=decode_payload(payload.read_bytes());V=d["vertices_f"][:,:3].astype(float);W=d["full_weights"].astype(float);F=d["indices"].reshape(-1,3)
 manifests=json.loads(lbs.MANIFEST.read_text());names={int(x["core_body_index"]):x["myosim_body"] for x in manifests["source"]["anchors"]}
 body=d["bindings_u"][:,0];A={};T={};native={};double={};case_receipts={}
 for case in ("reference","shoulder-elevation-050","elbow-flexion-080"):
  q=HERE/case;result=json.loads((q/"result.json").read_text());ci=json.loads((q/"invocation.json").read_text());snap=json.loads((q/"registered-skin-poses.json").read_text())
  if ci["binary"]!=inv["binary"] or ci["skin_payload_sha256"]!=inv["skin_payload_sha256"]:raise ValueError("unmatched source inputs")
  A[case],T[case]=affines(d,snap)
  double[case]=replay(V,W,A[case],T[case])
  data=np.genfromtxt(q/"registered-skin.csv",delimiter=",",skip_header=1)
  if data.shape!=(61950,20) or not np.array_equal(data[:,0],np.arange(len(V))):raise ValueError("invalid source CSV")
  native[case]=data[:,2:5]
  error=np.linalg.norm(double[case]-native[case],axis=1)
  # Snapshot serializer has 9 significant decimal digits; native uses Float32.
  if float(error.max())>2.5e-6:raise ValueError("source replay parity exceeded 2.5um: "+str(error.max()))
  case_receipts[case]={"max_native_replay_error_m":float(error.max()),"p95_native_replay_error_m":float(np.quantile(error,.95)),"csv_sha256":sha(q/"registered-skin.csv"),"pose_sha256":sha(q/"registered-skin-poses.json"),"native_exit_code":result["native_exit_code"],"physical_horizon_executed":False}
 cases={"shoulder-elevation-050":[79911,77840,12798],"elbow-flexion-080":[88675,80588,11508]}
 out={"source_payload_sha256":sha(payload),"script_sha256":sha(__file__),"observations":case_receipts,"boundary":"Current294 static native source registration only. Algebraic edge terms diagnose weight-gradient and within-binding geometry contributions; neither term alone proves causality or anatomical correctness. No asset changes, intersection test, skin/fat mechanics, dynamic ROM or contact qualification.","witnesses":{}}
 for case,faces in cases.items():
  rows=[]
  for face in faces:
   ids=F[face];lo,hi=stretch.principal_stretches(native["reference"][ids][None],native[case][ids][None])
   wm=W[ids];B0=np.einsum("vb,bij->vij",wm,A["reference"]);B1=np.einsum("vb,bij->vij",wm,A[case])
   local_singular=np.linalg.svd(B1@np.linalg.inv(B0),compute_uv=False)
   edges=[]
   for u,v in ((0,1),(1,2),(2,0)):
    i,j=int(ids[u]),int(ids[v]);rr={}
    for label in ("reference",case):
     g,w,terms=edge_decomposition(V[i],V[j],W[i],W[j],A[label],T[label])
     edge=double[label][j]-double[label][i]
     if not np.allclose(g+w,edge,atol=1e-12,rtol=1e-11):raise ValueError("edge decomposition not exact")
     top=np.argsort(np.linalg.norm(terms,axis=1))[-6:][::-1]
     rr[label]={"native_edge_length_m":float(np.linalg.norm(native[label][j]-native[label][i])),"double_replay_edge_m":edge.tolist(),"source_geometry_term_m":g.tolist(),"weight_gradient_term_m":w.tolist(),"term_norms_m":[float(np.linalg.norm(g)),float(np.linalg.norm(w))],"cancellation_ratio":float((np.linalg.norm(g)+np.linalg.norm(w))/max(np.linalg.norm(edge),1e-30)),"largest_weight_gradient_terms":[{"body_index":int(body[k]),"name":names[int(body[k])],"dw":float(W[j,k]-W[i,k]),"vector_m":terms[k].tolist()} for k in top]}
    edges.append({"vertices":[i,j],"weight_L1_jump":float(np.abs(W[j]-W[i]).sum()),"poses":rr})
   rows.append({"face":face,"vertices":ids.tolist(),"principal_surface_stretches":[float(lo[0]),float(hi[0])],"frozen_weight_local_relative_singular_values":local_singular.tolist(),"edges":edges})
  out["witnesses"][case]=rows
 dest=HERE/"current294-native-source-weight-decomposition.json"
 if dest.exists():raise ValueError("refuse overwrite")
 dest.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 for k,rows in out["witnesses"].items():
  for row in rows:print(k,row["face"],row["principal_surface_stretches"],"local-singular-range",float(np.min(row["frozen_weight_local_relative_singular_values"])),float(np.max(row["frozen_weight_local_relative_singular_values"])),flush=True)
if __name__=="__main__":main()
