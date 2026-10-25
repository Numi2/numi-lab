import importlib.util,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay,QhullError
base=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-layer-remesh-116")
out=base/"quality-repair-117"/"expanded-cavity-probe-002";out.mkdir(parents=True,exist_ok=False)
modp=base/"quality-repair-117"/"bistellar_connected_116.py"
sp=importlib.util.spec_from_file_location("m",modp);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
z=np.load(base/"run-001"/"candidate.npz");X=np.asarray(z["reference_points_m"],float);X32=np.asarray(z["points_world_f32"],np.float32).astype(float);T=np.asarray(z["tetrahedra"],int);mats=np.asarray(z["material_ids"],int);bad=[14022,23439]
faceinc=m.make_topology(T,X)[0];adj=[set() for _ in range(len(T))]
for f,incs in faceinc.items():
 if len(incs)==2:
  i,j=incs[0][0],incs[1][0]
  if mats[i]==mats[j]:adj[i].add(j);adj[j].add(i)
def find_inside(cent,old_tets):
 V=X[old_tets]; inv=[]; v0=[]
 for cell in V:
  A=np.column_stack((cell[1]-cell[0],cell[2]-cell[0],cell[3]-cell[0]))
  try: inv.append(np.linalg.inv(A))
  except np.linalg.LinAlgError: inv.append(None)
  v0.append(cell[0])
 inside=np.zeros(len(cent),bool)
 for a,b in zip(inv,v0):
  if a is None: continue
  ix=np.flatnonzero(~inside)
  if not len(ix):break
  q=(cent[ix]-b)@a.T;bary=np.column_stack((1-q.sum(axis=1),q))
  inside[ix]=np.min(bary,axis=1)>=-1e-9
 return inside
def run(level):
 cavity=set(bad);frontier=set(bad)
 for _ in range(level):
  nxt=set()
  for i in frontier:nxt.update(adj[i])
  nxt-=cavity;cavity|=nxt;frontier=nxt
 ids=np.array(sorted(cavity),int);old=T[ids];v_ids=np.unique(old);res={"ring":level,"cells":len(ids),"vertices":len(v_ids),"materials":sorted(set(mats[ids].tolist()))}
 if len(set(mats[ids].tolist()))!=1:res["status"]="mixed-material";return res
 if len(v_ids)>160 or len(ids)>400:res["status"]="cap-stop";return res
 try:tri=Delaunay(X[v_ids])
 except QhullError as e:res.update(status="delaunay-fail",error=str(e)[:120]);return res
 made=v_ids[tri.simplices];cent=X[made].mean(axis=1);made=made[find_inside(cent,old)]
 try:made=m.orient_ref(made,X)
 except Exception as e:res.update(status="orientation-fail",error=str(e));return res
 chain=m.boundary_chain(old,X)==m.boundary_chain(made,X)
 v0=float(m.dets(X,old).sum()/6);v1=float(m.dets(X,made).sum()/6)
 q=m.repair.cell_values_gradients(X32,X32,made)[2]; keep=np.ones(len(T),bool);keep[ids]=False;trial=np.vstack([T[keep],made]);qg=m.repair.cell_values_gradients(X32,X32,trial)[2]
 res.update({"delaunay_tets":len(tri.simplices),"inside_tets":len(made),"boundary_exact":bool(chain),"volume_before":v0,"volume_after":v1,"rel_volume_delta":(v1-v0)/v0,"local_q_min":float(q.min()),"local_q_bad":int(np.sum(q<1e-5)),"global_q_bad":int(np.sum(qg<1e-5)),"status":"pass_candidate" if chain and np.all(q>=1e-5) and np.all(m.dets(X32,made)>0) else ("boundary_match_q_fail" if chain else "boundary_mismatch")})
 return res
t=time.time();rows=[]
for n in (0,1,2,3):
 r=run(n);rows.append(r)
 if r.get("status") in ("pass_candidate","cap-stop"):break
rep={"input_path":str(base/"run-001"/"candidate.npz"),"input_sha256":hashlib.sha256((base/"run-001"/"candidate.npz").read_bytes()).hexdigest(),"target_bad_rows":bad,"same_material_face_neighbor_rings":rows,"seconds":time.time()-t,"status":"diagnostic_only_no_candidate"}
(out/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
print(json.dumps(rep,indent=2))
