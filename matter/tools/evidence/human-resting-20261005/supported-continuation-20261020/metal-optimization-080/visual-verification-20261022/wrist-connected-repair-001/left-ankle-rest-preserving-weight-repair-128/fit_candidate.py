from pathlib import Path
import sys,json,hashlib,time,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix,vstack,eye
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";P=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001")
sys.path.insert(0,str(R/"tools"));import rebind_human_skin_reference as rb
sys.path.insert(0,str(E/"joint-region-roi-014"));import run_ankle_tibia_weight_counterfactual_018 as wa
O=E/"left-ankle-rest-preserving-weight-repair-128";O.mkdir(exist_ok=False)
src=E/"bilateral-knee-connected-weight-repair-100/candidate.nhskin";p=rb.decode(src.read_bytes());F=p["inds"];W=p["weights"].astype(float)
wpath=E/"joint-region-roi-014/ankle-dorsiflexion-witnesses-expanded-017.json";old=rb.decode(wa.PAYLOAD.read_bytes());witness=json.loads(wpath.read_text())["witnesses"]
lookup={tuple(sorted(map(int,f))):i for i,f in enumerate(F)}
remap=[]
for w in witness:
 ids=np.array(w["source_vertices"]);assert set(map(int,old["inds"][w["source_skin_face"]]))==set(map(int,ids))
 fid=lookup[tuple(sorted(map(int,ids)))];assert np.array_equal(old["verts"][ids,:3],p["verts"][ids,:3])
 remap.append(dict(old_face=w["source_skin_face"],current_face=int(fid),source_vertices=ids.tolist()))
seed=np.unique([v for w in witness for v in w["source_vertices"]]+[39142,38765,39553]);incident=np.any(np.isin(F,seed),axis=1);sel=np.unique(F[incident]);sel=np.unique(F[np.any(np.isin(F,sel),axis=1)])
for _ in range(3):sel=np.unique(F[np.any(np.isin(F,sel),axis=1)])
by={int(x["body"]):i for i,x in enumerate(p["binds"])};ti,ca,ta=by[150],by[152],by[151];sel=sel[(W[sel,ti]+W[sel,ca])>.5]
faces=np.flatnonzero(np.any(np.isin(F,sel),axis=1));LF=F[faces];used=np.unique(LF);lmap={int(v):i for i,v in enumerate(sel)}
a0,b0=rb.reference_transforms(p,json.loads((P/"neutral.json").read_text()),[0,0,.35]);X=rb.positions(p,a0,b0)
folders={"reference":wa.CAP/"registered-reference-elevation-010-014","DF":wa.MATCH/"ankle-dorsiflexion-020-ref010","PF":wa.MATCH/"ankle-plantarflexion-020-ref010"}
cache={}
for name,folder in folders.items():
 snap=json.loads((folder/"registered-skin-poses.json").read_text());pose={"source_poses":[dict(core_body=x["body_index"],position=x["current"]["position_world_m"],quaternion_xyzw=x["current"]["orientation_world_xyzw"]) for x in snap["bodies"]]}
 a,b=rb.reference_transforms(p,pose,[0,0,0]);cache[name]=(a,b,snap,rb.positions(p,a,b))
tri=X[LF];A=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2);_,RR=np.linalg.qr(A,mode="reduced");invR=np.linalg.inv(RR);area=np.linalg.norm(np.cross(A[:,:,0],A[:,:,1]),axis=1)
# The recorded skin triangles agree with current emitted pose within Float32 error.
for w in witness:
 assert np.max(np.linalg.norm(cache["DF"][3][w["source_vertices"]]-np.array(w["skin_triangle_world_mm"])/1000,axis=1))<1e-6
normals=[];origins=[]
for w in witness:
 t=np.array(w["bone_triangle_world_mm"])/1000;n=np.cross(t[1]-t[0],t[2]-t[0]);n/=np.linalg.norm(n);normals.append(n);origins.append(t[0])
def evaluate(d):
 w=W[sel].copy();w[:,ti]+=d*(W[sel,ca]+W[sel,ta]);w[:,ca]-=d*W[sel,ca];w[:,ta]-=d*W[sel,ta]
 x=np.linalg.solve(np.einsum("nk,kij->nij",w,a0),(X[sel]-w@b0)[...,None])[...,0];out={}
 for name,(a,b,snap,Y0) in cache.items():
  Y=Y0.copy();Y[sel]=np.einsum("nk,kij,nj->ni",w,a,x)+w@b;out[name]=Y
 return out
edges=np.unique(np.sort(np.vstack((LF[:,[0,1]],LF[:,[1,2]],LF[:,[2,0]])),axis=1),axis=0);edges=[(lmap[int(i)],lmap[int(j)]) for i,j in edges if int(i) in lmap and int(j) in lmap]
def residual(d,summary=False):
 ys=evaluate(d);cons=[];metrics={}
 for name,Y in ys.items():
  tri=Y[LF];B=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2);ss=np.linalg.svd(B@invR,compute_uv=False);ar=np.linalg.norm(np.cross(B[:,:,0],B[:,:,1]),axis=1)/area
  cons.extend([np.minimum(ss[:,-1]-.35,0),np.minimum(2.4-ss[:,0],0),np.minimum(ar-.25,0)])
  metrics[name]={"min_stretch":float(ss[:,-1].min()),"max_stretch":float(ss[:,0].max()),"min_area":float(ar.min())}
 dist=np.array([(ys["DF"][w["source_vertices"]]-origins[i])@normals[i]*1000 for i,w in enumerate(witness)])
 if summary:return {"strain":metrics,"witness_clearance_mm_min":float(dist.min()),"witness_clearance_mm":dist.tolist()}
 return np.r_[np.minimum(dist.ravel()-.2,0),np.concatenate(cons)*10,1e-3*d,1e-2*np.array([d[i]-d[j] for i,j in edges])]


_,bonepath,brec,bv,bi,anchors=wa.audit.load_bones()
boneV,boneF,bowners=wa.audit.bone_scene(cache["DF"][2],"ankle","left",brec,bv,bi,anchors)
constraints={}
for w in witness:
 fid=lookup[tuple(sorted(w["source_vertices"]))];tri=np.array(w["bone_triangle_world_mm"])/1000;n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
 constraints[(fid,int(w["bone_face_local"]))]=(F[fid],n,tri[0],tri)
for dirname,part in [("left-ankle-rest-preserving-weight-repair-121","exact-report.json"),("left-ankle-rest-preserving-weight-repair-123/iteration-03","report.json")]:
 dd=E/dirname;z=np.load(dd/"solution.npz");rr=json.loads((dd/part).read_text());check=rr["poses"]["DF"] if "poses" in rr else rr["DF"];oldfaces=z["current_faces"]
 for group in check["patch_vs_left_ankle_bones"]["witnesses_by_owner_pair"].values():
  for h in group:
   fid=int(oldfaces[h["face_a"]]);bid=int(h["face_b"])-len(oldfaces);tri=boneV[boneF[bid]].astype(float);n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n);constraints[(fid,bid)]=(F[fid],n,tri[0],tri)
constraints=list(constraints.values())

clear=rb.decode((E/"left-ankle-rest-preserving-weight-repair-125/iteration-00/candidate.nhskin").read_bytes());Yclear=rb.positions(clear,cache["DF"][0],cache["DF"][1])
branches=[]
for ids,_,_,tri in constraints:
 A=Yclear[ids];B=tri
 ea=np.roll(A,-1,axis=0)-A;eb=np.roll(B,-1,axis=0)-B
 na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1]);axes=[na,nb]
 axes.extend(np.cross(u,v) for u in ea for v in eb)
 axes.extend(np.cross(na,u) for u in ea);axes.extend(np.cross(nb,u) for u in eb)
 choices=[]
 for axis in axes:
  length=np.linalg.norm(axis)
  if length<1e-14:continue
  axis=axis/length
  for n in (axis,-axis):
   origin=n*np.max(B@n);gap=float(np.min(A@n)-np.max(B@n));choices.append((gap,n,origin))
 gap,n,origin=max(choices,key=lambda z:z[0]);branches.append((ids,n,origin))
print("SAT clear minimum mm",min(float(np.min((Yclear[i]-o)@n))*1000 for i,n,o in branches),flush=True)
constraints=branches

ncon=3*len(constraints);nqual=len(cache)*3*len(LF);nr=ncon+nqual+len(sel)+len(edges)
sp=lil_matrix((nr,len(sel)),dtype=int)
for j,(ids,_,_) in enumerate(constraints):
 for k,v in enumerate(ids):
  if int(v) in lmap:sp[3*j+k,lmap[int(v)]]=1
for pi in range(len(cache)):
 for fi,face in enumerate(LF):
  for v in face:
   if int(v) in lmap:
    for component in range(3):sp[ncon+pi*3*len(LF)+component*len(LF)+fi,lmap[int(v)]]=1
sp[ncon+nqual:ncon+nqual+len(sel),:]=eye(len(sel))
for j,(i,k) in enumerate(edges):sp[ncon+nqual+len(sel)+j,i]=1;sp[ncon+nqual+len(sel)+j,k]=1
def res(d,summary=False):
 yy=evaluate(d);cons=[];metrics={}
 for name,Y in yy.items():
  tri=Y[LF];B=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2);ss=np.linalg.svd(B@invR,compute_uv=False);ar=np.linalg.norm(np.cross(B[:,:,0],B[:,:,1]),axis=1)/area
  cons.extend([np.minimum(ss[:,-1]-.65,0),np.minimum(1.5-ss[:,0],0),np.minimum(ar-.5,0)])
  metrics[name]={"min_stretch":float(ss[:,-1].min()),"max_stretch":float(ss[:,0].max()),"min_area":float(ar.min())}
 distances=np.concatenate([(yy["DF"][ids]-org)@n*1000 for ids,n,org in constraints])
 if summary:return {"strain":metrics,"constraint_min_mm":float(distances.min())}
 return np.r_[np.minimum(distances-.01,0),np.concatenate(cons)*10,.001*d,.02*np.array([d[i]-d[j] for i,j in edges])]
initial=np.zeros(len(sel));saved=np.load(E/"left-ankle-rest-preserving-weight-repair-125/iteration-00/solution.npz");initial[np.searchsorted(sel,saved["selected"])]=saved["delta"]
print("prepared",len(sel),len(LF),len(constraints),nr,flush=True)
t=time.monotonic();sol=least_squares(res,initial,bounds=(np.zeros(len(sel)),np.ones(len(sel))*.999),jac_sparsity=sp.tocsr(),method="trf",tr_solver="lsmr",max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
weights=W.copy();weights[sel,ti]+=sol.x*(W[sel,ca]+W[sel,ta]);weights[sel,ca]-=sol.x*W[sel,ca];weights[sel,ta]-=sol.x*W[sel,ta];blob,rr=rb.rebind(p,weights,a0,b0);cp=rb.decode(blob);(O/"candidate.nhskin").write_bytes(blob);np.savez(O/"solution.npz",selected=sel,delta=sol.x,current_faces=faces)
report={"metrics":res(sol.x,True),"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"candidate_sha256":hashlib.sha256(blob).hexdigest(),"rebind":rr,"variables":len(sel),"incident_faces":len(LF),"nfev":sol.nfev,"success":bool(sol.success),"seconds":time.monotonic()-t,"physical_admission":False}
(O/"fit-report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
