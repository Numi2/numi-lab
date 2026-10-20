#!/usr/bin/env python3
"""Source-bound regional ligament path audit; no endpoint/enthesis validation implied."""
import json, hashlib, math, struct, sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/numi-human-aortic-contact-opt-20261019")
sys.path.insert(0,str(ROOT/"tools"))
import audit_registered_ligament_geometry as ag
INVOCATION=Path("/Users/numi/numi-human-aortic-contact-opt-20261019/build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-003/invocation.json")
# Broad local-frame patches are explicit search regions, not source-defined entheses.
PATCHES={
 "ATFL_fibula_to_talus": ((20,22),(10,-3,25),(16,-3,12),((0,-12,18),(22,5,32)),((6,-14,6),(25,2,20))),
 "CFL_fibula_to_calcaneus": ((20,6),(-8,-4,25),(-15,-25,17),((-20,-15,18),(5,3,33)),((-25,-37,8),(-4,-10,24))),
 "PTFL_fibula_to_talus": ((20,22),(-12,-4,25),(-12,-4,12),((-25,-12,18),(0,5,33)),((-25,-15,6),(2,5,22))),
 "deltoid_anterior_tibiotalar": ((4,22),(12,13,-20),(12,0,-12),((3,3,-31),(24,23,-12)),((3,-10,-23),(25,8,-6))),
 "deltoid_tibiocalcaneal": ((4,6),(0,12,-20),(0,-20,-15),((-10,3,-31),(10,23,-12)),((-20,-35,-25),(15,-10,-5))),
 "deltoid_posterior_tibiotalar": ((4,22),(-5,18,-20),(-12,-3,-12),((-23,8,-31),(5,25,-12)),((-25,-12,-23),(5,7,-6))),
}
def rot(q,v):
 xyz=np.asarray(q[:3],dtype=np.float64)
 return v+q[3]*2*np.cross(xyz,v)+2*np.cross(xyz,np.cross(xyz,v))
def norm_rows(v):
 n=np.linalg.norm(v,axis=1);return v/np.maximum(n[:,None],1e-20)
def load_surface_normals(sid,frame,bb,bp,records,vertex0,index0):
 r=records[sid]
 raw=np.frombuffer(bb,dtype="<f4",count=r["vertexCount"]*6,offset=vertex0+24*r["firstVertex"]).reshape(-1,6)
 ns=raw[:,3:].astype(np.float64)
 ns=norm_rows(np.asarray([ag.qinvrot(frame["q"],ag.qrot(bp[r["body"]][3:],ag.qrot(r["boneQ"],n))) for n in ns]))
 inds=struct.unpack_from(f"<{r['indexCount']}I",bb,index0+4*r["firstIndex"])
 tris=np.asarray(inds,dtype=np.int32).reshape(-1,3)-r["firstVertex"]
 return norm_rows(ns[tris].mean(axis=1))
def regional_candidates(mesh, normals, center_mm, bounds_mm, limit=48):
 v=np.asarray(mesh[0],dtype=np.float64);tris=np.asarray(mesh[1],dtype=np.int32);pts=v[tris].mean(axis=1)
 center=np.asarray(center_mm,dtype=np.float64)/1000
 lo=np.asarray(bounds_mm[0],dtype=np.float64)/1000;hi=np.asarray(bounds_mm[1],dtype=np.float64)/1000
 ids=np.flatnonzero(np.all((pts>=lo)&(pts<=hi),axis=1))
 ids=ids[np.argsort(np.sum((pts[ids]-center)**2,axis=1))[:limit]]
 return [(pts[i],int(i),float(np.linalg.norm(pts[i]-center)),normals[i]) for i in ids]
def vector_hits(a,b,mesh):
 verts=np.asarray(mesh[0],dtype=np.float64);tris=np.asarray(mesh[1],dtype=np.int32)
 A=verts[tris[:,0]];B=verts[tris[:,1]];C=verts[tris[:,2]]
 d=b-a;e1=B-A;e2=C-A
 # Cull triangle AABBs before Moller-Trumbore.
 seglo=np.minimum(a,b)-1e-9;seghi=np.maximum(a,b)+1e-9
 broad=np.all(np.maximum(np.minimum(A,np.minimum(B,C)),seglo)<=np.minimum(np.maximum(A,np.maximum(B,C)),seghi),axis=1)
 idx=np.flatnonzero(broad)
 if not len(idx):return []
 A=A[idx];B=B[idx];C=C[idx];e1=e1[idx];e2=e2[idx]
 h=np.cross(np.broadcast_to(d,e2.shape),e2);det=np.einsum("ij,ij->i",e1,h)
 scale=np.linalg.norm(d)*np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)
 ok=np.abs(det)>32*np.finfo(np.float64).eps*scale
 inv=np.zeros_like(det);inv[ok]=1/det[ok];ss=a-A;u=inv*np.einsum("ij,ij->i",ss,h);q=np.cross(ss,e1)
 vv=inv*np.einsum("j,ij->i",d,q);t=inv*np.einsum("ij,ij->i",e2,q)
 mask=ok&(u>=-1e-8)&(u<=1+1e-8)&(vv>=-1e-8)&(u+vv<=1+1e-8)&(t>1e-5)&(t<1-1e-5)
 return [(float(ti),int(idx[j])) for j,ti in enumerate(t) if mask[j]]
def all_meshes(frame,bb,bodypose,records,vertex0,index0):
 out={}
 for sid,r in records.items():
  if r["body"] not in bodypose: continue
  raw=np.frombuffer(bb,dtype="<f4",count=r["vertexCount"]*6,offset=vertex0+24*r["firstVertex"]).reshape(-1,6)
  xyz=raw[:,:3].astype(np.float64)*r["scale"]
  xyz=np.asarray([ag.qrot(r["boneQ"],v) for v in xyz])+np.asarray(r["translation"])
  bp=bodypose[r["body"]];world=np.asarray([ag.qrot(bp[3:],v) for v in xyz])+np.asarray(bp[:3])
  local=np.asarray([ag.qinvrot(frame["q"],v-np.asarray(frame["origin"])) for v in world])
  inds=struct.unpack_from(f"<{r['indexCount']}I",bb,index0+4*r["firstIndex"])
  tris=np.asarray(inds,dtype=np.int32).reshape(-1,3)-r["firstVertex"]
  out[sid]=(local,tris)
 return out
def mesh_topology(tris):
 edges=np.vstack((tris[:,[0,1]],tris[:,[1,2]],tris[:,[2,0]]));edges.sort(axis=1)
 _,cnt=np.unique(edges,axis=0,return_counts=True)
 return {"boundaryEdges":int(np.sum(cnt==1)),"nonManifoldEdges":int(np.sum(cnt>2)),"closed2Manifold":bool(np.all(cnt==2))}
def mesh_tuple(v,t): return (v.tolist(),t.tolist())
def main():
 invb=INVOCATION.read_bytes();inv=json.loads(invb);argv=inv["argv"]
 if Path(argv[1])!=ag.RIGID or Path(argv[3])!=ag.BONES: raise RuntimeError("payload paths are not exact native-supported-003 argv[1]/argv[3]")
 rb,bb,rh,bh,js,dq,bodypose,records,_,vertex0,index0=ag.read_payloads()
 missing_pose=sorted(sid for sid,r in records.items() if r["body"] not in bodypose)
 if missing_pose: raise RuntimeError(f"registered bone records lack source pose: {missing_pose}")
 source={"boneRecordsWithSourcePose":len(records),"recordsMissingSourcePose":missing_pose,"invocationPath":str(INVOCATION),"invocationSha256":hashlib.sha256(invb).hexdigest(),"sourceInvocationSha256":inv.get("source_sha256"),
  "rigidPath":argv[1],"rigidSha256":hashlib.sha256(rb).hexdigest(),"bonesPath":argv[3],"bonesSha256":hashlib.sha256(bb).hexdigest(),
  "boneCount":bh[2],"vertexCount":bh[3],"indexCount":bh[4],"catalogHeaderSha256":hashlib.sha256((ROOT/"apps/NumiHumanRegisteredLigaments.hpp").read_bytes()).hexdigest()}
 out={"audit":"source_bound_regional_path_audit_v1","classification":"inferred regional proxy paths; no subject enthesis validation or ligament coverage claim","source":source,"criteria":{"patches":"explicit local-frame boxes and proxy centers in mm","candidate":"source triangle centroid rounded only when serialized to float32 anchors","outwardNormalCosMinimum":0.15,"segmentIntersection":"scale-relative Moller-Trumbore; interior t in (1e-5,1-1e-5); triangle AABB cull","meshes":"all 185 registered source records, with endpoint meshes separated in result; open meshes prevent closed-volume proof"},"paths":[]}
 for group,(right_ids,c0r,c1r,b0r,b1r) in PATCHES.items():
  for side in (0,1):
   sign=1 if side==1 else -1
   c0=np.array((c0r[0],c0r[1],sign*c0r[2]),float)/1000;c1=np.array((c1r[0],c1r[1],sign*c1r[2]),float)/1000
   bounds0=(np.array(b0r[0],float),np.array(b0r[1],float));bounds1=(np.array(b1r[0],float),np.array(b1r[1],float))
   if side==0:
    bounds0[0][2],bounds0[1][2]=-b0r[1][2],-b0r[0][2];bounds1[0][2],bounds1[1][2]=-b1r[1][2],-b1r[0][2]
   s0,s1=right_ids
   if side==0:s0={20:21,4:5}[s0];s1={22:23,6:7}[s1]
   parent,child=((150,151) if side==0 else (136,137));frame=ag.make_frame(js,dq,bodypose,parent,child)
   surface0=ag.build_mesh(s0,frame,records,bb,vertex0,index0,bodypose);surface1=ag.build_mesh(s1,frame,records,bb,vertex0,index0,bodypose)
   normals0=load_surface_normals(s0,frame,bb,bodypose,records,vertex0,index0);normals1=load_surface_normals(s1,frame,bb,bodypose,records,vertex0,index0)
   cand0=regional_candidates(surface0,normals0,c0*1000,bounds0);cand1=regional_candidates(surface1,normals1,c1*1000,bounds1)
   allm=all_meshes(frame,bb,bodypose,records,vertex0,index0)
   if len(allm)!=len(records): raise RuntimeError(f"transformed {len(allm)} of {len(records)} registered source bone meshes")
   pairs=sorted(((a[2]**2+b[2]**2+0.1*np.linalg.norm(a[0]-b[0])**2,a,b) for a in cand0 for b in cand1),key=lambda p:p[0])
   result=None;tested=0
   for score,a,b in pairs:
    tested+=1
    # Float32 anchors are the values the native catalog can actually store.
    p0=np.asarray(a[0],dtype=np.float32).astype(np.float64);p1=np.asarray(b[0],dtype=np.float32).astype(np.float64);delta=p1-p0;length=np.linalg.norm(delta)
    if not length: continue
    cos0=float(np.dot(a[3],delta)/length);cos1=float(np.dot(b[3],-delta)/length)
    if cos0<0.15 or cos1<0.15: continue
    nearby=[]
    seglo=np.minimum(p0,p1)-1e-9;seghi=np.maximum(p0,p1)+1e-9
    for sid,(vv,tt) in allm.items():
     if np.all(np.max(vv,axis=0)>=seglo) and np.all(np.min(vv,axis=0)<=seghi):nearby.append(sid)
    h=[]
    for sid in nearby:
     hits=vector_hits(p0,p1,mesh_tuple(*allm[sid]))
     if hits:h.append({"stableId":sid,"sourceRecord":records[sid]["sourceRecord"],"body":records[sid]["body"],"hits":[{"t":round(t,8),"triangle":ti} for t,ti in hits]})
    if not h:
     result=(a,b,p0,p1,cos0,cos1,nearby);break
   entry={"allRegisteredBoneMeshesTested":len(allm),"group":group,"side":"R" if side else "L","stableIds":[s0,s1],"sourceRecords":[records[s0]["sourceRecord"],records[s1]["sourceRecord"]],"frameJointPair":[parent,child],
    "candidateCounts":[len(cand0),len(cand1)],"testedPairs":tested,"endpointMeshesTopology":{"surface0":mesh_topology(np.asarray(surface0[1],dtype=np.int32)),"surface1":mesh_topology(np.asarray(surface1[1],dtype=np.int32))}}
   if result:
    a,b,p0,p1,cos0,cos1,nearby=result
    def closest_on_tri(point,mesh,ti):
     tri=mesh[1][ti];v=mesh[0];q=ag.closest(tuple(point),v[tri[0]],v[tri[1]],v[tri[2]])
     return float(np.linalg.norm(point-np.asarray(q)))
    entry.update({"accepted":True,"anchor0LocalFloat32M":[float(x) for x in p0],"anchor1LocalFloat32M":[float(x) for x in p1],"anchor0LocalMm":[round(float(x*1000),3) for x in p0],"anchor1LocalMm":[round(float(x*1000),3) for x in p1],
     "sourceTriangleIds":[a[1],b[1]],"centerResidualMm":[round(a[2]*1000,3),round(b[2]*1000,3)],"float32AnchorToWitnessTriangleResidualMm":[round(closest_on_tri(p0,surface0,a[1])*1000,6),round(closest_on_tri(p1,surface1,b[1])*1000,6)],
     "lengthMm":round(float(np.linalg.norm(p1-p0)*1000),3),"outwardNormalCos":[round(cos0,4),round(cos1,4)],"aabbNearbyStableIds":nearby,"segmentTriangleHits":[]})
   else:entry.update({"accepted":False,"reason":"no candidate pair passed outward normal and all-mesh triangle checks"})
   out["paths"].append(entry)
 out["sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 path=ROOT/"tools/audit_ligament_regional_paths.json";path.write_text(json.dumps(out,indent=2)+"\n")
 print("receipt",path,"sha256",out["sha256"],"paths",len(out["paths"]))
 for e in out["paths"]:print(e["group"],e["side"],"accepted",e["accepted"],"nearby",e.get("aabbNearbyStableIds"),"hits",e.get("segmentTriangleHits"),"len",e.get("lengthMm"),"normal",e.get("outwardNormalCos"),"res",e.get("float32AnchorToWitnessTriangleResidualMm"))
if __name__=="__main__":main()
