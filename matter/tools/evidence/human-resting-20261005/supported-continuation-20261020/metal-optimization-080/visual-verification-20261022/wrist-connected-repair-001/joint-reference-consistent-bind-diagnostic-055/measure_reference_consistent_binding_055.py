#!/usr/bin/env python3
"""Numerical registration counterfactual; never updates scene assets or physics."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent; E=HERE.parent
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module("decomp",E/"joint-source-decomposition-053/measure_source_weight_decomposition_053.py")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(A0,t0,A1,t1):
 R=A1@np.linalg.inv(A0); return R,t1-np.einsum("bij,bj->bi",R,t0)
def rebased(P,W,R,t):
 return m.replay(P,W,R,t)
# Independent invariants: identity, rigid covariance, single influence, no state mutation.
rng=np.random.default_rng(555)
P=rng.normal(size=(25,3));W=rng.random((25,3));W/=W.sum(axis=1)[:,None]
I=np.tile(np.eye(3),(3,1,1));z=np.zeros((3,3))
assert np.allclose(rebased(P,W,I,z),P,atol=1e-14,rtol=1e-14)
Q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);t=np.array([3.,-2.,5.])
assert np.allclose(rebased(P,W,np.tile(Q,(3,1,1)),np.tile(t,(3,1))),P@Q.T+t,atol=1e-14,rtol=1e-14)
V=P.copy();one=np.zeros_like(W);one[:,1]=1
assert np.allclose(rebased(P,one,np.tile(Q,(3,1,1)),np.tile(t,(3,1))),P@Q.T+t)
assert np.array_equal(V,P)
inv=json.loads((E/"joint-source-decomposition-053/reference/invocation.json").read_text())
argv=inv["argv"];payload=Path(argv[argv.index("--skin-payload")+1])
assert sha(payload)=="d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04"
d=m.decode_payload(payload.read_bytes());V=d["vertices_f"][:,:3].astype(float)
W0=d["full_weights"].astype(float);W=W0/W0.sum(axis=1)[:,None];F=d["indices"].reshape(-1,3)
roi=m.stretch.roi
names={int(x["core_body_index"]):x["myosim_body"] for x in json.loads(roi.MAN.read_text())["source"]["anchors"]}
owners=[names[int(x)] for x in d["bindings_u"][:,0][W.argmax(axis=1)]]
regions=np.asarray([roi.majority([roi.skin_region(owners[int(i)]) for i in f]) for f in F])
sides=np.asarray([roi.majority([roi.side(owners[int(i)]) for i in f]) for f in F])
base=E/"joint-source-decomposition-053/reference"
A0,t0=m.affines(d,json.loads((base/"registered-skin-poses.json").read_text()))
P0=m.replay(V,W,A0,t0)
out={"source_payload_sha256":sha(payload),"script_sha256":sha(__file__),
 "self_tests":"PASS identity rigid_covariance single_influence input_immutability",
 "formula":"P(q)=sum_b normalized(w_b) [A_b(q) inverse(A_b(q0)) P(q0) + t_b(q)-A_b(q) inverse(A_b(q0)) t_b(q0)]",
 "baseline":"Current294 declared source preparation pose; not a claim of measured neutral anatomy.",
 "weight_normalization_max_change":float(np.abs(W-W0).max()),
 "boundary":"Static preparation counterfactual only. No asset, binding payload, material, force, controller, FEM transfer or accepted-state owner modified. Not a skin mechanics fix, motion acceptance, intersection test or ROM qualification.",
 "cases":{}}
cases=[("shoulder-elevation-050","shoulder",E/"joint-source-decomposition-053"),("elbow-flexion-080","elbow",E/"joint-source-decomposition-053"),
 ("ankle-dorsiflexion-020","ankle",E/"current294-joint-source-check-052"),("ankle-plantarflexion-020","ankle",E/"current294-joint-source-check-052"),
 ("mtp-flexion-020","mtp",E/"current294-joint-source-check-052"),("mtp-extension-020","mtp",E/"current294-joint-source-check-052")]
arrays={"reference":P0}
for case,joint,parent in cases:
 q=parent/case;snap=q/"registered-skin-poses.json"
 if not snap.exists():
  out["cases"][case]={"unmeasured":"matching all-body pose snapshot absent"};continue
 A,t=m.affines(d,json.loads(snap.read_text()));R,tr=relative(A0,t0,A,t)
 # Every relative map must be rigid: the source registration scale cancels.
 singular=np.linalg.svd(R,compute_uv=False)
 assert np.max(np.abs(singular-1))<1e-6
 old=m.replay(V,W,A,t);new=rebased(P0,W,R,tr)
 arrays[case]=new
 lo0,hi0=m.stretch.principal_stretches(P0[F],old[F]);lo1,hi1=m.stretch.principal_stretches(P0[F],new[F])
 groups=roi.GROUPS[joint] if joint in roi.GROUPS else {"foot"}
 rr={}
 for side in ("left","right"):
  ids=np.flatnonzero((sides==side)&np.isin(regions,list(groups)))
  worst=ids[np.argsort(lo1[ids])[:6]]
  rr[side]={"faces":len(ids),"original_min_stretch":m.stretch.stats(lo0[ids]),"counterfactual_min_stretch":m.stretch.stats(lo1[ids]),
   "original_max_stretch":m.stretch.stats(hi0[ids]),"counterfactual_max_stretch":m.stretch.stats(hi1[ids]),
   "counterfactual_witnesses":[{"face":int(i),"vertices":F[i].tolist(),"stretches":[float(lo1[i]),float(hi1[i])]} for i in worst]}
 out["cases"][case]={"all_relative_maps_max_singular_error":float(np.abs(singular-1).max()),"max_position_change_m":float(np.linalg.norm(new-old,axis=1).max()),"pose_sha256":sha(snap),"sides":rr}
 print(case,{s:(rr[s]["original_min_stretch"]["min"],rr[s]["counterfactual_min_stretch"]["min"],rr[s]["counterfactual_max_stretch"]["max"]) for s in rr},flush=True)
np.savez_compressed(HERE/"counterfactual-positions.npz",**arrays)
out["positions_sha256"]=sha(HERE/"counterfactual-positions.npz")
(HERE/"reference-consistent-bind-diagnostic.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
