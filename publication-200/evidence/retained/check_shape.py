from pathlib import Path
import numpy as np,json,sys
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638");O=Path(__file__).parent
ps=[S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz",S/"per-bank-workers-20261027/b124-123/root-layer-self-barrier-007/candidate-private.npz",O/"candidate.npz"]
d=[]
for p in ps:
 with np.load(p,allow_pickle=False) as z:d.append({k:z[k] for k in z.files})
x,base,y=[r["points_world_m"].astype(float) for r in d];f=d[0]["source_O_faces"]
assert set(d[0])==set(d[2]) and all(d[0][k].tobytes()==d[2][k].tobytes() for k in d[0] if k!="points_world_m")
def q(u,v):
 ids=np.flatnonzero(np.any(u!=v,axis=1));fs=np.flatnonzero(np.any(np.isin(f,ids),axis=1));a=u[f[fs]];b=v[f[fs]];n=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);m=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);l=np.linalg.norm(n,axis=1);r=np.linalg.norm(m,axis=1);c=np.sum(n*m,axis=1)/(l*r)
 angles=[]
 for T in (a,b):
  arr=[]
  for i in range(3):
   u=T[:,(i+1)%3]-T[:,i];w=T[:,(i+2)%3]-T[:,i]
   arr.append(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i",u,w)/(np.linalg.norm(u,axis=1)*np.linalg.norm(w,axis=1)),-1,1))))
  angles.append(np.min(np.array(arr),axis=0))
 return {"changed_nodes":len(ids),"changed_faces":len(fs),"min_area_ratio":float((r/l).min()),"max_area_ratio":float((r/l).max()),"min_normal_cos":float(c.min()),"worst_min_angle_change_degrees":float((angles[1]-angles[0]).min())}
o={"versus_prior":q(base,y),"versus_original":q(x,y),"metadata_equal":True,"admitted":False}
o["shape_gate_pass"]=o["versus_original"]["min_area_ratio"]>=.75 and o["versus_original"]["max_area_ratio"]<=1.3 and o["versus_original"]["min_normal_cos"]>=.9 and o["versus_original"]["worst_min_angle_change_degrees"]>=-5.
(O/"shape-comparison.json").write_text(json.dumps(o,indent=2)+"\n");print(json.dumps(o));sys.exit(0 if o["shape_gate_pass"] else 1)
