from pathlib import Path
import hashlib,json,numpy as np
D=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=D/"guarded-61-002/skin-fat-layers-541.npz";c=D/"guarded-61-007/skin-fat-layers-541.npz"
assert sha(b)=="65732a1c2efe231766bf690d81ad343602547175a34853cfb4a4364bd5fe688e"
assert sha(c)=="591d106026479969ed5757ef1a87a11e11004f81533296c03acb8ca3a96376ec"
def vol(a,b,c,d):return np.einsum("ij,ij->i",np.cross(b-a,c-a),d-a)/6
def analyze(z,outerkey,innerkey,fieldkey):
 outer=z[outerkey];inner=z[innerkey];src=z["outer_faces"];f=np.sort(src,axis=1);a,b,c=f.T
 inv=((src[:,0]>src[:,1]).astype(np.int8)+(src[:,0]>src[:,2]).astype(np.int8)+(src[:,1]>src[:,2]).astype(np.int8));parity=inv&1
 v0=vol(outer[a],outer[b],outer[c],inner[c]);v1=vol(outer[a],outer[b],inner[b],inner[c]);v2=vol(outer[a],inner[a],inner[b],inner[c])
 field=z[fieldkey];tol=np.maximum(1e-8,np.maximum.reduce([field[a],field[b],field[c]]))**3*1e-12;expected=np.where(parity,1.,-1.)
 authored=np.isfinite(v0)&np.isfinite(v1)&np.isfinite(v2)&(np.abs(v0)>tol)&(np.abs(v1)>tol)&(np.abs(v2)>tol)&(np.sign(v0)==expected)&(np.sign(v1)==-expected)&(np.sign(v2)==expected)
 ids=np.stack([np.stack([a,b,c,len(outer)+c],axis=1),np.stack([a,b,len(outer)+b,len(outer)+c],axis=1),np.stack([a,len(outer)+a,len(outer)+b,len(outer)+c],axis=1)],axis=1).reshape(-1,4)
 xyz64=np.concatenate((outer,inner));q=xyz64[ids];v64=vol(q[:,0],q[:,1],q[:,2],q[:,3]);reverse=v64<0;ids[reverse,0],ids[reverse,1]=ids[reverse,1].copy(),ids[reverse,0].copy()
 xyz32=xyz64.astype(np.float32).astype(np.float64);p=xyz32[ids];v=vol(p[:,0],p[:,1],p[:,2],p[:,3]);long=np.zeros(len(p))
 for i in range(4):
  for j in range(i+1,4):long=np.maximum(long,np.linalg.norm(p[:,i]-p[:,j],axis=1))
 nd=6*v/np.maximum(long,1e-300)**3
 return {"authored_fail":set(map(int,np.flatnonzero(~authored))),"f32_nonpositive":set(map(int,np.flatnonzero(v<=0))),"f32_conditioned_fail":set(map(int,np.flatnonzero(nd<1e-5)))}
result={}
with np.load(b,allow_pickle=False) as x,np.load(c,allow_pickle=False) as y:
 assert np.array_equal(x["outer_faces"],y["outer_faces"])
 for label,ok,ik,fk in [("skin","outer_points_f64","inner_points_f64","skin_thickness_transferred_m"),("fat","inner_points_f64","deep_points_f64","fat_thickness_transferred_m")]:
  x0=analyze(x,ok,ik,fk);x1=analyze(y,ok,ik,fk);result[label]={}
  for name in x0:
   result[label][name]={"before":len(x0[name]),"after":len(x1[name]),"added":sorted(x1[name]-x0[name]),"removed":sorted(x0[name]-x1[name])}
report={"status":"DIAGNOSTIC_NO_PHYSICAL_ADMISSION","baseline_sha256":sha(b),"candidate_sha256":sha(c),"per_layer":result,"caution":"Orientation and conditioning checks are diagnostics for the inherited offset-prism construction; no physical mass, global layer collision, or native admission is established."}
(D/"guarded-61-007-cell-comparison.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
