#!/usr/bin/env python3
"""Reference-exact DQS counterfactual, frozen current294 captures only. No asset edits."""
from pathlib import Path
import importlib.util,json,hashlib,numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;E=HERE.parent
spec=importlib.util.spec_from_file_location("source053",E/"joint-source-decomposition-053/measure_source_weight_decomposition_053.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mul(a,b):
 return np.concatenate((a[...,:3]*b[...,3,None]+b[...,:3]*a[...,3,None]+np.cross(a[...,:3],b[...,:3]),(a[...,3]*b[...,3]-np.sum(a[...,:3]*b[...,:3],axis=-1))[...,None]),axis=-1)
def conj(q):return q*np.array([-1,-1,-1,1])
def rotate(q,v):return v+2*np.cross(q[...,:3],np.cross(q[...,:3],v)+q[...,3,None]*v)
def blend(A,t,W):
 scale=np.cbrt(np.linalg.det(A));assert np.ptp(scale)<2e-6 and np.all(scale>0)
 R=A/scale[:,None,None];assert np.max(np.abs(R@R.transpose(0,2,1)-np.eye(3)))<2e-6
 q=Rotation.from_matrix(R).as_quat();qd=.5*mul(np.c_[t,np.zeros(len(t))],q)
 anchor=q[W.argmax(axis=1)];dots=anchor@q.T;sign=np.where(dots<0,-1.,1.);weights=W*sign
 qr=weights@q;dual=weights@qd;norm=np.linalg.norm(qr,axis=1);assert norm.min()>.1
 qr/=norm[:,None];dual/=norm[:,None];dual-=qr*np.sum(qr*dual,axis=1)[:,None]
 trans=2*mul(dual,conj(qr))[:,:3]
 return qr,trans,{"minimum_blend_norm":float(norm.min()),"weighted_hemisphere_margin_min":float(np.min(np.abs(dots)[W>1e-7]))}
def stats(x):return {str(k):float(v) for k,v in zip([0,1,50,95,99,100],np.percentile(x,[0,1,50,95,99,100]))}
def main():
 inv=json.loads((E/"joint-source-decomposition-053/reference/invocation.json").read_text());a=inv["argv"];payload=Path(a[a.index("--skin-payload")+1]);assert sha(payload)=="d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04"
 d=m.decode_payload(payload.read_bytes());V=d["vertices_f"][:,:3].astype(float);W=d["full_weights"].astype(float);F=d["indices"].reshape(-1,3)
 roi=m.stretch.roi;names={int(x["core_body_index"]):x["myosim_body"] for x in json.loads(roi.MAN.read_text())["source"]["anchors"]};owners=[names[int(x)] for x in d["bindings_u"][:,0][W.argmax(axis=1)]]
 regions=np.array([roi.majority([roi.skin_region(owners[int(i)]) for i in t]) for t in F]);sides=np.array([roi.majority([roi.side(owners[int(i)]) for i in t]) for t in F])
 refdir=E/"joint-source-decomposition-053/reference";ref=m.stretch.roi.read_csv(refdir/"registered-skin.csv").astype(float)
 snap=json.loads((refdir/"registered-skin-poses.json").read_text());A,t=m.affines(d,snap);q0,t0,bs=blend(A,t,W);local=rotate(conj(q0),ref-t0)
 assert np.max(np.abs(rotate(q0,local)+t0-ref))<1e-12
 # Rigid/single-owner algebra and antipodal invariance are exact to double tolerance.
 q=np.tile(Rotation.from_rotvec([.1,-.2,.3]).as_quat(),(3,1));p=np.arange(9).reshape(3,3)*.01
 assert np.max(np.abs(rotate(conj(q),rotate(q,p))-p))<1e-14
 assert np.max(np.abs(rotate(-q,p)-rotate(q,p)))<1e-14
 out={"payload_sha256":sha(payload),"script_sha256":sha(__file__),"reference_csv_sha256":sha(refdir/"registered-skin.csv"),"reference_pose_sha256":sha(refdir/"registered-skin-poses.json"),"reference_exact_error_m":float(np.max(np.abs(rotate(q0,local)+t0-ref))),"scope":"CPU counterfactual only; no source geometry, physics, forces, admission, or native presentation changed. Reference-exact DQS motion retains current294 reference positions. No collision, layer or anatomical qualification; cannot be promoted from strain metrics alone.","reference_blend":bs,"cases":{}}
 cases={"shoulder-elevation-050":("shoulder",roi.GROUPS["shoulder"]),"elbow-flexion-080":("elbow",roi.GROUPS["elbow"]),"knee-flexion-025":("knee",roi.GROUPS["knee"]),"ankle-dorsiflexion-020":("ankle",roi.GROUPS["ankle"]),"ankle-plantarflexion-020":("ankle",roi.GROUPS["ankle"]),"mtp-flexion-020":("mtp",{"foot"}),"mtp-extension-020":("mtp",{"foot"})}
 for case,(joint,groups) in cases.items():
  candidates=[E/"joint-source-decomposition-053"/case,E/"current294-joint-source-check-052"/case];folder=next((x for x in candidates if (x/"registered-skin.csv").exists() and (x/"registered-skin-poses.json").exists()),None)
  if folder is None:out["cases"][case]={"unmeasured":True};continue
  sn=json.loads((folder/"registered-skin-poses.json").read_text())
  if len(sn["bodies"])!=86:out["cases"][case]={"unmeasured":True,"reason":"capture lacks all86bindingposes"};continue
  A,t=m.affines(d,sn);qc,tc,b=blend(A,t,W);target=rotate(qc,local)+tc;native=roi.read_csv(folder/"registered-skin.csv").astype(float)
  parity=np.linalg.norm(m.replay(V,W,A,t)-native,axis=1);assert max(parity)<2.5e-6
  baseLo,baseHi=m.stretch.principal_stretches(ref[F],native[F]);lo,hi=m.stretch.principal_stretches(ref[F],target[F]);caseout={"joint":joint,"pose_sha256":sha(folder/"registered-skin-poses.json"),"csv_sha256":sha(folder/"registered-skin.csv"),"native_LBS_replay_max_m":float(max(parity)),"blend":b,"sides":{}}
  for side in ["left","right"]:
   ids=np.flatnonzero((sides==side)&np.isin(regions,list(groups)));worst=ids[np.argsort(lo[ids])[:5]]
   caseout["sides"][side]={"faces":len(ids),"lbs_min_stretch":stats(baseLo[ids]),"dqs_min_stretch":stats(lo[ids]),"lbs_max_stretch":stats(baseHi[ids]),"dqs_max_stretch":stats(hi[ids]),"lbs_area_ratio":stats((baseLo*baseHi)[ids]),"dqs_area_ratio":stats((lo*hi)[ids]),"dqs_worst_faces":[{"face":int(k),"lo":float(lo[k]),"hi":float(hi[k])} for k in worst]}
  out["cases"][case]=caseout
  print(case,{side:[round(z["lbs_min_stretch"]["0"],5),round(z["dqs_min_stretch"]["0"],5)] for side,z in caseout["sides"].items()},flush=True)
 dest=HERE/"joint-dqs-counterfactual.json";assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
