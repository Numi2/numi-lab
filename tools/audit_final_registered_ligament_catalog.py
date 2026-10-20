#!/usr/bin/env python3
"""Audit the exact per-side seed data in NumiHumanRegisteredLigaments.hpp against native-supported-003 source geometry."""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/numi-human-aortic-contact-opt-20261019")
sys.path.insert(0,str(ROOT/"tools"))
import audit_registered_ligament_geometry as ag
import audit_ligament_regional_paths as lp
HEADER=ROOT/"apps/NumiHumanRegisteredLigaments.hpp"
INVOCATION=Path("/Users/numi/numi-human-aortic-contact-opt-20261019/build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-003/invocation.json")
CATALOG={
 "iliofemoral_lateral":(19,18,3,2,"hip"),
 "ischiofemoral":(19,18,3,2,"hip"),
 "iliofemoral_medial":(19,18,3,2,"hip"),
 "pubofemoral":(19,18,3,2,"hip"),
 "superior_glenohumeral_capsuloligament":(11,10,13,12,"shoulder"),
 "coracohumeral_capsuloligament":(11,10,13,12,"shoulder"),
 "anterior_talofibular_superior":(21,20,23,22,"ankle"),
 "anterior_talofibular_inferior":(21,20,23,22,"ankle"),
 "calcaneofibular":(21,20,7,6,"ankle"),
 "posterior_talofibular":(21,20,23,22,"ankle"),
 "anterior_tibiotalar":(5,4,23,22,"ankle"),
 "tibiocalcaneal":(5,4,7,6,"ankle"),
 "posterior_tibiotalar":(5,4,23,22,"ankle"),
 "elbow_rcl":(13,12,17,16,"elbow"),
 "coracoclavicular_complex":(11,10,9,8,"cc"),
 "lumbar_interspinous_L1_L2":(164,164,165,165,"lumbar"),
 "lumbar_interspinous_L2_L3":(165,165,166,166,"lumbar"),
 "lumbar_interspinous_L3_L4":(166,166,167,167,"lumbar"),
 "lumbar_interspinous_L4_L5":(167,167,168,168,"lumbar"),
}
def parse_vec(s):return tuple(float(x.strip()) for x in s.split(","))
def parse_header():
 text=HEADER.read_text()
 witness={}
 pat=re.compile(r'if\(n=="([^"]+)"\) witness\(side==0\?V3\{([^}]*)\}:V3\{([^}]*)\},side==0\?V3\{([^}]*)\}:V3\{([^}]*)\}\);')
 midpat=re.compile(r'else if\(n=="([^"]+)"\) witness\(V3\{([^}]*)\},V3\{([^}]*)\}\);')
 for line in text.splitlines():
  m=pat.search(line)
  if m:
   name=m.group(1)
   if name in witness:raise ValueError(f"duplicate witness branch {name}")
   witness[name]={0:(parse_vec(m.group(2)),parse_vec(m.group(4))),1:(parse_vec(m.group(3)),parse_vec(m.group(5)))}
   continue
  mid=midpat.search(line)
  if mid:
   name=mid.group(1)
   if name in witness:raise ValueError(f"duplicate witness branch {name}")
   witness[name]={2:(parse_vec(mid.group(2)),parse_vec(mid.group(3)))}
 adds=set(re.findall(r'\badd\("([^"]+)"',text))
 if adds!=set(witness):raise ValueError(f"catalog add/witness mismatch add={sorted(adds)} witness={sorted(witness)}")
 if not adds<=set(CATALOG):raise ValueError(f"unmapped final catalog entries: {sorted(adds-set(CATALOG))}")
 return text,witness,adds
def frame_pair(region,side,name=None):
 if region=="hip":return (144,145) if side==0 else (130,131)
 if region=="shoulder":return (90,91) if side==0 else (40,41)
 if region=="elbow":return (91,92) if side==0 else (41,42)
 if region=="cc":return (83,84) if side==0 else (33,34)
 if region=="lumbar":
  return {"lumbar_interspinous_L1_L2":(15,16),"lumbar_interspinous_L2_L3":(12,13),
          "lumbar_interspinous_L3_L4":(9,10),"lumbar_interspinous_L4_L5":(3,4)}[name]
 return (150,151) if side==0 else (136,137)
def topology(tris):return lp.mesh_topology(np.asarray(tris,dtype=np.int32))
def load_vertex_normals(sid,frame,bb,bp,records,vertex0):
 r=records[sid]
 raw=np.frombuffer(bb,dtype="<f4",count=r["vertexCount"]*6,offset=vertex0+24*r["firstVertex"]).reshape(-1,6)
 ns=raw[:,3:].astype(np.float64)
 ns=np.asarray([ag.qinvrot(frame["q"],ag.qrot(bp[r["body"]][3:],ag.qrot(r["boneQ"],n))) for n in ns])
 lengths=np.linalg.norm(ns,axis=1)
 return ns/np.maximum(lengths[:,None],1e-30)
def interpolated_normal(mesh,vertex_normals,triangle_id,point):
 ids=np.asarray(mesh[1][triangle_id],dtype=np.int64)
 a,b,c=np.asarray(mesh[0][ids],dtype=np.float64);p=np.asarray(point,dtype=np.float64)
 ab=b-a;ac=c-a;ap=p-a
 d00=float(np.dot(ab,ab));d01=float(np.dot(ab,ac));d11=float(np.dot(ac,ac));d20=float(np.dot(ap,ab));d21=float(np.dot(ap,ac))
 den=d00*d11-d01*d01
 if abs(den)<1e-24:raise RuntimeError("degenerate source triangle in projected normal")
 wb=(d11*d20-d01*d21)/den;wc=(d00*d21-d01*d20)/den;wa=1.0-wb-wc
 n=wa*vertex_normals[ids[0]]+wb*vertex_normals[ids[1]]+wc*vertex_normals[ids[2]]
 length=float(np.linalg.norm(n))
 if not np.isfinite(length) or length<=1e-30:raise RuntimeError("invalid interpolated source normal")
 return n/length

def main():
 header_text,witness,add_names=parse_header();header_hash=hashlib.sha256(header_text.encode()).hexdigest()
 invb=INVOCATION.read_bytes();inv=json.loads(invb);argv=inv["argv"]
 if Path(argv[1])!=ag.RIGID or Path(argv[3])!=ag.BONES:raise RuntimeError("source paths do not match exact native-supported-003 argv[1]/argv[3]")
 rb,bb,rh,bh,js,dq,bodypose,records,_,vertex0,index0=ag.read_payloads()
 missing=sorted(sid for sid,r in records.items() if r["body"] not in bodypose)
 if missing:raise RuntimeError(f"source body pose missing for stable IDs {missing}")
 source={"headerPath":str(HEADER),"headerSha256":header_hash,"invocationPath":str(INVOCATION),"invocationFileSha256":hashlib.sha256(invb).hexdigest(),"sourceInvocationSha256":inv.get("source_sha256"),
 "rigidPath":argv[1],"rigidSha256":hashlib.sha256(rb).hexdigest(),"bonesPath":argv[3],"bonesSha256":hashlib.sha256(bb).hexdigest(),"boneAbi":bh[1],"boneCount":bh[2],"vertexCount":bh[3],"indexCount":bh[4],"boneRecordsWithSourcePose":len(records),"recordsMissingSourcePose":missing}
 out={"audit":"final_registered_ligament_catalog_source_geometry_v1","classification":"reference-pose admission audit of inferred regional anchors; no subject-specific enthesis validation","source":source,
 "criteria":{"catalog":"per-side seed triples parsed directly from the hashed header, no candidate patch target substitution","endpoint":"project seed to registered stableId mesh; reproduce body-local float32 GPU anchor then back-transform at reference pose","normal":"transformed source vertex normals barycentrically interpolated at the projected point, matching NumiHumanInferredACL.hpp; outward cosine >=-1e-4 numerical non-inward guard required","segment":"scale-relative Moller-Trumbore with AABB culling against every registered bone source mesh; interior t in (1e-5,1-1e-5)","topology":"surface meshes may be open; triangle-clearance does not prove closed-volume containment"},"rows":[]}
 allmesh_cache={};normal_cache={}
 for name in sorted(add_names):
  s0L,s0R,s1L,s1R,region=(*CATALOG[name][:4],CATALOG[name][4])
  sides=(2,) if region=="lumbar" else (0,1)
  for side in sides:
   s0,s1=(s0L,s1L) if side in (0,2) else (s0R,s1R)
   if s0 not in records or s1 not in records:raise RuntimeError(f"missing stable ID for {name} side {side}")
   parent,child=frame_pair(region,side,name);fk=(parent,child)
   frame=ag.make_frame(js,dq,bodypose,parent,child)
   if fk not in allmesh_cache:
    allm=lp.all_meshes(frame,bb,bodypose,records,vertex0,index0)
    if len(allm)!=len(records):raise RuntimeError(f"transformed {len(allm)}/{len(records)} bones")
    allmesh_cache[fk]=(frame,allm)
   frame,allm=allmesh_cache[fk]
   m0=allm[s0];m1=allm[s1]
   if (s0,fk) not in normal_cache:normal_cache[(s0,fk)]=load_vertex_normals(s0,frame,bb,bodypose,records,vertex0)
   if (s1,fk) not in normal_cache:normal_cache[(s1,fk)]=load_vertex_normals(s1,frame,bb,bodypose,records,vertex0)
   seed0=np.asarray(witness[name][side][0],dtype=np.float64);seed1=np.asarray(witness[name][side][1],dtype=np.float64)
   p0,d0,t0=ag.project(tuple(seed0),m0);p1,d1,t1=ag.project(tuple(seed1),m1)
   # Projected frame-local points are the native cooker endpoints before body-local float32 storage.
   body0=records[s0]["body"];body1=records[s1]["body"]
   def local_from_frame(point,body):
    world=ag.add(frame["origin"],ag.qrot(frame["q"],point));pose=bodypose[body]
    return ag.qinvrot(pose[3:],ag.sub(world,pose[:3]))
   def frame_from_local(local,body):
    pose=bodypose[body];world=ag.add(pose[:3],ag.qrot(pose[3:],local))
    return ag.qinvrot(frame["q"],ag.sub(world,frame["origin"]))
   gpuLocal0=np.asarray(local_from_frame(p0,body0),dtype=np.float32).astype(np.float64)
   gpuLocal1=np.asarray(local_from_frame(p1,body1),dtype=np.float32).astype(np.float64)
   actual0=np.asarray(frame_from_local(gpuLocal0,body0),dtype=np.float64);actual1=np.asarray(frame_from_local(gpuLocal1,body1),dtype=np.float64)
   normal0=interpolated_normal(m0,normal_cache[(s0,fk)],t0,p0);normal1=interpolated_normal(m1,normal_cache[(s1,fk)],t1,p1)
   delta=actual1-actual0;length=float(np.linalg.norm(delta))
   cos0=float(np.dot(normal0,delta)/length) if length>0 else -1
   cos1=float(np.dot(normal1,-delta)/length) if length>0 else -1
   seglo=np.minimum(actual0,actual1)-1e-9;seghi=np.maximum(actual0,actual1)+1e-9;near=[];hits=[]
   for sid,(vv,tt) in allm.items():
    if np.all(np.max(vv,axis=0)>=seglo) and np.all(np.min(vv,axis=0)<=seghi):
     near.append(sid);hh=lp.vector_hits(actual0,actual1,lp.mesh_tuple(vv,tt))
     if hh:hits.append({"stableId":sid,"sourceRecord":records[sid]["sourceRecord"],"body":records[sid]["body"],"hits":[{"t":round(t,8),"triangle":ti} for t,ti in hh]})
   f32d0=ag.project(tuple(actual0),m0)[1];f32d1=ag.project(tuple(actual1),m1)[1]
   row={"name":name,"side":"M" if side==2 else ("L" if side==0 else "R"),"stableIds":[s0,s1],"sourceRecords":[records[s0]["sourceRecord"],records[s1]["sourceRecord"]],"endpointBodies":[body0,body1],"frameJointPair":[parent,child],
    "headerSeed0FrameM":[float(x) for x in seed0],"headerSeed1FrameM":[float(x) for x in seed1],"seedProjectionResidualMm":[round(d0*1000,6),round(d1*1000,6)],"sourceProjectionTriangles":[t0,t1],
    "gpuLocalAnchor0Float32M":[float(x) for x in gpuLocal0],"gpuLocalAnchor1Float32M":[float(x) for x in gpuLocal1],"float32ReconstructedFrameAnchor0Mm":[round(float(x*1000),6) for x in actual0],"float32ReconstructedFrameAnchor1Mm":[round(float(x*1000),6) for x in actual1],
    "float32AnchorSurfaceResidualMm":[round(f32d0*1000,6),round(f32d1*1000,6)],"lengthMm":round(length*1000,4),"outwardNormalCos":[round(cos0,5),round(cos1,5)],"aabbNearbyStableIds":near,"allRegisteredBoneMeshesTested":len(allm),"segmentTriangleHits":hits,
    "endpointMeshTopology":{"surface0":topology(m0[1]),"surface1":topology(m1[1])}}
   row["admittedByAudit"]=bool(max(d0,d1)<=0.02 and min(cos0,cos1)>=-1e-4 and not hits and max(f32d0,f32d1)<=1e-5)
   out["rows"].append(row)
 if hashlib.sha256(HEADER.read_bytes()).hexdigest()!=header_hash:raise RuntimeError("catalog header changed during final audit")
 out["sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 path=ROOT/"tools/audit_final_registered_ligament_catalog.json";path.write_text(json.dumps(out,indent=2)+"\n")
 print("receipt",path,"sha256",out["sha256"],"rows",len(out["rows"]),"alladmitted",sum(x["admittedByAudit"] for x in out["rows"]))
 for r in out["rows"]:print(r["name"],r["side"],"admit",r["admittedByAudit"],"res",r["seedProjectionResidualMm"],"tri",r["sourceProjectionTriangles"],"len",r["lengthMm"],"normal",r["outwardNormalCos"],"near",r["aabbNearbyStableIds"],"hits",r["segmentTriangleHits"],"f32res",r["float32AnchorSurfaceResidualMm"])
if __name__=="__main__":main()
