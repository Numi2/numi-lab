#!/usr/bin/env python3
"""One bounded exact-pose search for posterior lumbar rows that lengthen in flexion."""
import argparse,json,hashlib,sys,math
from pathlib import Path
import numpy as np
sys.path.insert(0,"tools")
import audit_registered_ligament_geometry as ag
import audit_actual_pose_registered_ligament_catalog as ap
import audit_ligament_regional_paths as lp
import audit_final_registered_ligament_catalog as base
import audit_lumbar_interspinous_candidates as old
ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--invocation",type=Path,default=ROOT/"build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-support-multipoint-048/invocation.json")
parser.add_argument("--output",type=Path,default=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/lumbar-interspinous-052/audit_lumbar_interspinous_flexion_candidates_048.json")
args=parser.parse_args();INV=args.invocation.resolve();OUT=args.output.resolve()
invb=INV.read_bytes();inv=json.loads(invb);argv=inv["argv"]
ag.RIGID=Path(argv[1]);ag.BONES=Path(argv[3])
rb,bb,rh,bh,joints,dq,src,records,_,vertex0,index0=ag.read_payloads();q=list(map(float,dq))
for i,x in enumerate(argv):
 if x=="--pose-q":q[int(argv[i+1])]=float(argv[i+2])
nh=Path(argv[argv.index("--joint-equality-payload")+1]);nheb,neh,q,eqcorr=ap.parse_nheq(nh,q)
baseposes,_,_=ap.forward_fk(joints,src,q)
qflex=q.copy();qflex[7]-=.05;nheb,neh,qflex,flexcorr=ap.parse_nheq(nh,qflex)
flexposes,_,_=ap.forward_fk(joints,src,qflex)
segments=[
 {"name":"interspinous_L1_L2","cranialBody":19,"cranialStableId":164,"caudalBody":16,"caudalStableId":165,"framePair":[15,16],"frameJoint":15},
 {"name":"interspinous_L2_L3","cranialBody":16,"cranialStableId":165,"caudalBody":13,"caudalStableId":166,"framePair":[12,13],"frameJoint":12},
 {"name":"interspinous_L3_L4","cranialBody":13,"cranialStableId":166,"caudalBody":10,"caudalStableId":167,"framePair":[9,10],"frameJoint":9},
 {"name":"interspinous_L4_L5","cranialBody":10,"cranialStableId":167,"caudalBody":4,"caudalStableId":168,"framePair":[3,4],"frameJoint":3},
]
def body_local(pt,frame,poses,body):
 w=frame["origin"]+ap.qrot(frame["q"],pt);bp=poses[body]
 return ap.qrot([-bp[3],-bp[4],-bp[5],bp[6]],w-bp[:3])
def body_world(local,poses,body):
 bp=poses[body];return bp[:3]+ap.qrot(bp[3:],local)
def frame_point(world,frame):
 return ap.qrot([-frame["q"][0],-frame["q"][1],-frame["q"][2],frame["q"][3]],world-frame["origin"])
def rotation_vector(q):
 q=ap.qnorm(q)
 if q[3]<0:q=-q
 s=float(np.linalg.norm(q[:3]))
 if s<1e-15:return [0.0,0.0,0.0]
 return (q[:3]*(2.0*math.atan2(s,float(q[3]))/s)).tolist()
def path_hits(p0,p1,meshes,records):
 lo=np.minimum(p0,p1)-1e-9;hi=np.maximum(p0,p1)+1e-9;near=[];hits=[]
 for sid,(vv,tt) in meshes.items():
  if np.all(np.max(vv,axis=0)>=lo) and np.all(np.min(vv,axis=0)<=hi):
   near.append(sid);hh=lp.vector_hits(p0,p1,lp.mesh_tuple(vv,tt))
   if hh:hits.append({"stableId":sid,"body":records[sid]["body"],"hits":[[float(t),int(ti)] for t,ti in hh]})
 return near,hits
out={"audit":"lumbar_interspinous_flexion_alternative_search_v1","classification":"single bounded inferred-region candidate search; not exact enthesis validation or subject calibration","source":{"searchScriptPath":str(Path(__file__).resolve()),"searchScriptSha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"baseMasterQ7Rad":q[7],"flexMasterQ7Rad":qflex[7],"invocationPath":str(INV),"invocationSha256":hashlib.sha256(invb).hexdigest(),"sourceInvocationSha256":inv.get("source_sha256"),"binarySha256":inv.get("binary_sha256"),"rigidPath":str(ag.RIGID),"rigidSha256":hashlib.sha256(rb).hexdigest(),"bonesPath":str(ag.BONES),"bonesSha256":hashlib.sha256(bb).hexdigest(),"nheqPath":str(nh),"nheqSha256":hashlib.sha256(nheb).hexdigest(),"boneAbi":bh[1],"boneRecordCount":bh[2],"vertexCount":bh[3],"indexCount":bh[4],"flexionMasterQ":7,"flexionDeltaRad":-0.05,"baseNheqMaxCorrection":eqcorr,"flexNheqMaxCorrection":flexcorr},"criteria":{"regions":"posterior 18th-percentile triangle centroids, central |Z|<=18 mm, adjacent edge quartiles; 64 candidates per bone surface","flexion":"NHEQ-projected q7 -0.05 rad; confirm local cranial-vs-caudal sagittal relative rotation is negative around local +Z (anterior flexion under +X anterior,+Y superior,+Z lateral)","lengthening":"endpoint-material length increase must exceed 0.01 mm at 5-deg flexion probe","facing":"base and flex projected source triangle normals are non-inward within numerical tolerance; base cosines >=0.05","path":"straight segment has no scale-relative segment/triangle hit against all 185 source bone meshes at base or flex pose","limitations":"registered meshes are open; no closed-solid nonpenetration proof, only proximal lumbar L1-L5 covered; exact entheses not measured"},"segments":[]}
cache={}
for seg in segments:
 ji,fr0=ap.body_frame(joints,baseposes,tuple(seg["framePair"]))
 jf,frf=ap.body_frame(joints,flexposes,tuple(seg["framePair"]))
 if ji!=seg["frameJoint"] or jf!=ji:raise RuntimeError("frame identity changed "+seg["name"])
 bodyc,bodyd=seg["cranialBody"],seg["caudalBody"]
 qci0=ap.qmul([-fr0["q"][0],-fr0["q"][1],-fr0["q"][2],fr0["q"][3]],baseposes[bodyc][3:])
 qdi0=ap.qmul([-fr0["q"][0],-fr0["q"][1],-fr0["q"][2],fr0["q"][3]],baseposes[bodyd][3:])
 qcif=ap.qmul([-frf["q"][0],-frf["q"][1],-frf["q"][2],frf["q"][3]],flexposes[bodyc][3:])
 qdif=ap.qmul([-frf["q"][0],-frf["q"][1],-frf["q"][2],frf["q"][3]],flexposes[bodyd][3:])
 rel0=ap.qmul([-qdi0[0],-qdi0[1],-qdi0[2],qdi0[3]],qci0)
 relf=ap.qmul([-qdif[0],-qdif[1],-qdif[2],qdif[3]],qcif)
 rel_delta=ap.qmul(relf,[-rel0[0],-rel0[1],-rel0[2],rel0[3]])
 rot0=rotation_vector(rel0);rotf=rotation_vector(relf);rotd=rotation_vector(rel_delta)
 joint_rotation={"cranialBody":bodyc,"caudalBody":bodyd,"frameJoint":ji,"definition":"relative quaternion = inverse(caudal orientation in the instantaneous entry-joint frame) * cranial orientation in that frame; flex delta = flex relative quaternion * inverse(base relative quaternion)","relativeRotationVectorBaseRad":rot0,"relativeRotationVectorFlexRad":rotf,"relativeRotationChangeVectorRad":rotd,"sagittalChangeAboutFrameLocalZDeg":rotd[2]*180.0/math.pi,"frameAxes":"+X anterior, +Y superior, +Z lateral; negative local-Z rotation tilts the cranial superior axis anterior (flexion)"}
 if ji not in cache:
  meshes0=lp.all_meshes(fr0,bb,{i:x for i,x in enumerate(baseposes) if x is not None},records,vertex0,index0)
  meshesf=lp.all_meshes(frf,bb,{i:x for i,x in enumerate(flexposes) if x is not None},records,vertex0,index0)
  cache[ji]=(meshes0,meshesf,fr0,frf)
 meshes0,meshesf,fr0,frf=cache[ji]
 sid0,sid1=seg["cranialStableId"],seg["caudalStableId"]
 m0,m1=meshes0[sid0],meshes0[sid1];mf0,mf1=meshesf[sid0],meshesf[sid1]
 nrm0=base.load_vertex_normals(sid0,fr0,bb,{i:x for i,x in enumerate(baseposes) if x is not None},records,vertex0)
 nrm1=base.load_vertex_normals(sid1,fr0,bb,{i:x for i,x in enumerate(baseposes) if x is not None},records,vertex0)
 nrm0f=base.load_vertex_normals(sid0,frf,bb,{i:x for i,x in enumerate(flexposes) if x is not None},records,vertex0)
 nrm1f=base.load_vertex_normals(sid1,frf,bb,{i:x for i,x in enumerate(flexposes) if x is not None},records,vertex0)
 nt0=old.tri_normals(m0,nrm0);nt1=old.tri_normals(m1,nrm1);nt0f=old.tri_normals(mf0,nrm0f);nt1f=old.tri_normals(mf1,nrm1f)
 c0,info0=old.candidates(m0,nt0,True,64);c1,info1=old.candidates(m1,nt1,False,64)
 body0=records[sid0]["body"];body1=records[sid1]["body"]
 loc0=[body_local(c[0],fr0,baseposes,body0) for c in c0];loc1=[body_local(c[0],fr0,baseposes,body1) for c in c1]
 flex0=[frame_point(body_world(x,flexposes,body0),frf) for x in loc0];flex1=[frame_point(body_world(x,flexposes,body1),frf) for x in loc1]
 pairs=[]
 for ia,a in enumerate(c0):
  for ib,b in enumerate(c1):
   length=float(np.linalg.norm(b[0]-a[0]));lf=float(np.linalg.norm(flex1[ib]-flex0[ia]));delta=(lf-length)*1000
   score=a[2]**2+b[2]**2+.02*np.linalg.norm(a[0]-b[0])**2
   if .002<=length<=.060 and delta>.01:pairs.append((score,delta,length,ia,ib))
 pairs.sort(key=lambda x:x[0]);tested=0;accepted=None;rejected=[]
 for score,delta,length,ia,ib in pairs:
  tested+=1;a,b=c0[ia],c1[ib];p0,p1=a[0],b[0];axis=(p1-p0)/length
  f0=float(np.dot(a[3],axis));f1=float(np.dot(b[3],-axis))
  if min(f0,f1)<.05:continue
  af0,af1=flex0[ia],flex1[ib];flexLength=float(np.linalg.norm(af1-af0));axisf=(af1-af0)/max(flexLength,1e-30)
  fn0=float(np.dot(nt0f[a[1]],axisf));fn1=float(np.dot(nt1f[b[1]],-axisf))
  if min(fn0,fn1)<-1e-4:continue
  near0,hits0=path_hits(p0,p1,meshes0,records)
  if hits0:continue
  nearf,hitsf=path_hits(af0,af1,meshesf,records)
  if hitsf:continue
  accepted={"anchorsFrameMm":[(p0*1000).tolist(),(p1*1000).tolist()],"sourceTriangles":[a[1],b[1]],"targetResidualMm":[a[2]*1000,b[2]*1000],"referenceLengthMm":length*1000,"flexLengthMm":flexLength*1000,"flexLengthDeltaMm":delta,"q7FlexionDeltaRad":-.05,"outwardNormalCosBase":[f0,f1],"outwardNormalCosFlex":[fn0,fn1],"baseNearbyStableIds":near0,"flexNearbyStableIds":nearf,"baseHits":hits0,"flexHits":hitsf,"testedCandidatesBeforeAdmission":tested,"targetScore":float(score)}
  break
 outseg={**seg,"allRegisteredBoneMeshesTested":len(meshes0),"candidateCounts":[len(c0),len(c1)],"regionCranial":info0,"regionCaudal":info1,"relativeJointRotation":joint_rotation,"flexionLengtheningCandidates":len(pairs),"candidatePairsCheckedUntilAdmission":tested,"accepted":accepted is not None,"witness":accepted,"endpointTopology":[lp.mesh_topology(np.asarray(m0[1],dtype=np.int32)),lp.mesh_topology(np.asarray(m1[1],dtype=np.int32))]}
 out["segments"].append(outseg)
 print(seg["name"],"candidates",len(pairs),"accepted",accepted is not None,"pairs_checked",tested,"witness",accepted)
out["sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
outpath=OUT;outpath.parent.mkdir(parents=True,exist_ok=True);outpath.write_text(json.dumps(out,indent=2)+"\n")
print("receipt",outpath,"sha256",out["sha256"],"admitted",sum(x["accepted"] for x in out["segments"]),"of",len(out["segments"]))
