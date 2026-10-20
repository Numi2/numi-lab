#!/usr/bin/env python3
"""Exact-invocation, NHEQ-projected source-geometry audit for ligament seeds at a commanded pose."""
import argparse,hashlib,json,math,re,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"tools"))
import audit_registered_ligament_geometry as ag
import audit_ligament_regional_paths as lp
import audit_final_registered_ligament_catalog as base
HEADER=ROOT/"apps/NumiHumanRegisteredLigaments.hpp"
INV=None
def qmul(a,b):
 x,y,z,w=a;X,Y,Z,W=b
 return (w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z)
def qrot(q,v):
 q=np.asarray(q,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
 u=np.cross(q[:3],v)
 return v+2*q[3]*u+2*np.cross(q[:3],u)
def qnorm(q):
 a=np.asarray(q,dtype=np.float64);n=np.linalg.norm(a)
 if not np.isfinite(n) or n<=0:raise ValueError("invalid quaternion")
 return a/n
def parse_nheq(path,q):
 raw=path.read_bytes();h=struct.unpack_from("<8s10I32s",raw,0)
 if h[0]!=b"NHEQ1\0\0\0" or h[1]!=1 or h[5]!=96:raise ValueError("unsupported NHEQ1")
 count=h[4]
 if len(raw)!=80+count*96:raise ValueError("NHEQ1 dimensions mismatch")
 out=list(map(float,q));maxcorr=0.
 for i in range(count):
  row=struct.unpack_from("<4I20f",raw,80+96*i)
  dep,depv,master,masterv=row[:4]
  y0,x0,a0,a1,a2,a3,a4=row[4:11]
  delta=0. if master==0xffffffff else out[master]-x0
  target=y0+a0+delta*(a1+delta*(a2+delta*(a3+delta*a4)))
  maxcorr=max(maxcorr,abs(out[dep]-target));out[dep]=target
 return raw,h,out,maxcorr
def forward_fk(joints,bodypose,q):
 n=max(max(j[:2]) for j in joints)+1
 poses=[None]*n;children={j[1] for j in joints};roots=[i for i in range(n) if i not in children]
 for r in roots:
  if r in bodypose:poses[r]=np.asarray(bodypose[r],dtype=np.float64)
 pending=list(enumerate(joints));loops=0
 for loops in range(n+2):
  rest=[];changed=False
  for ji,j in pending:
   pb,cb=j[:2]
   if poses[pb] is None:rest.append((ji,j));continue
   z=j[8:];typ=j[2];qoff=j[4];nq=j[5]
   axis=np.asarray(z[0:3],dtype=np.float64);pa=np.asarray(z[12:15],dtype=np.float64);ca=np.asarray(z[16:19],dtype=np.float64)
   pr=qnorm(z[20:24]);cr=qnorm(z[24:28]);motion=np.array([0.,0.,0.,1.])
   if typ in (0,2) and nq:
    axis=axis/np.linalg.norm(axis);theta=q[qoff]
    motion=np.r_[axis*math.sin(theta/2),math.cos(theta/2)]
   elif typ==1 and nq:
    axis=axis/np.linalg.norm(axis);motion[:3]=0.
    # A prismatic coordinate shifts along parent-joint axis; accounted below.
   rp=poses[pb][3:]
   rot=qnorm(qmul(qmul(qmul(rp,pr),motion),[-cr[0],-cr[1],-cr[2],cr[3]]))
   parentOffset=qrot(rp,pa)
   if typ==1 and nq: parentOffset=parentOffset+qrot(qmul(rp,pr),axis)*q[qoff]
   pos=poses[pb][:3]+parentOffset-qrot(rot,ca)
   poses[cb]=np.r_[pos,rot];changed=True
  pending=rest
  if not changed:break
 if pending:raise RuntimeError(f"FK unresolved joints {len(pending)}")
 return poses,roots,loops+1
def body_frame(joints,poses,pair):
 found=[(i,j) for i,j in enumerate(joints) if j[0]==pair[0] and j[1]==pair[1]]
 if len(found)!=1:raise ValueError(f"joint pair {pair} occurrences={len(found)}")
 idx,j=found[0];z=j[8:];parent_pose=poses[j[0]];child_pose=poses[j[1]]
 parent_anchor=parent_pose[:3]+qrot(parent_pose[3:],z[12:15])
 child_anchor=child_pose[:3]+qrot(child_pose[3:],z[16:19])
 residual=float(np.linalg.norm(parent_anchor-child_anchor))
 if not np.isfinite(residual) or residual>.0002:raise RuntimeError(f"joint frame anchor residual exceeds native 0.2 mm guard: {pair} {residual}")
 return idx,{"origin":.5*(parent_anchor+child_anchor),"q":qnorm(qmul(parent_pose[3:],qnorm(z[20:24]))),"anchorResidualM":residual,"parent":j[0],"child":j[1]}
def frame_info(name,side):
 if name in {"iliofemoral_lateral","ischiofemoral","iliofemoral_medial","pubofemoral"}:return ((144,145) if side==0 else (130,131))
 if "glenohumeral" in name or "coracohumeral" in name:return ((90,91) if side==0 else (40,41))
 if name=="elbow_rcl":return ((91,92) if side==0 else (41,42))
 return ((150,151) if side==0 else (136,137))
def stable_pair(name,side):
 if name in {"iliofemoral_lateral","ischiofemoral","iliofemoral_medial","pubofemoral"}:return (19,3) if side==0 else (18,2)
 if "glenohumeral" in name or "coracohumeral" in name:return (11,13) if side==0 else (10,12)
 if name=="elbow_rcl":return (13,17) if side==0 else (12,16)
 if name in {"anterior_talofibular_superior","anterior_talofibular_inferior","posterior_talofibular"}:return (21,23) if side==0 else (20,22)
 if name=="calcaneofibular":return (21,7) if side==0 else (20,6)
 if name in {"anterior_tibiotalar","posterior_tibiotalar"}:return (5,23) if side==0 else (4,22)
 if name=="tibiocalcaneal":return (5,7) if side==0 else (4,6)
 raise ValueError(f"unmapped catalog name {name}")
def main():
 global INV,HEADER
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--invocation",type=Path,required=True,help="exact native invocation.json")
 parser.add_argument("--header",type=Path,default=HEADER,help="registered ligament header to audit")
 parser.add_argument("--output",type=Path,default=None,help="JSON receipt path (default: tools/audit_actual_pose_registered_ligament_catalog.json)")
 parser.add_argument("--native-stderr",type=Path,default=None,help="optional runtime stderr for quantitative cook-diagnostic comparison")
 args=parser.parse_args()
 INV=args.invocation.resolve();HEADER=args.header.resolve()
 base.HEADER=HEADER
 invb=INV.read_bytes();inv=json.loads(invb);argv=inv["argv"]
 if len(argv)<4:raise RuntimeError("invocation argv does not contain rigid and bone payload paths")
 ag.RIGID=Path(argv[1]);ag.BONES=Path(argv[3])
 if not ag.RIGID.is_file() or not ag.BONES.is_file():raise RuntimeError("invocation payload path is missing")
 getarg=lambda flag: Path(argv[argv.index(flag)+1])
 rb,bb,rh,bh,joints,dq,sourcePose,records,_,vertex0,index0=ag.read_payloads()
 q=list(map(float,dq));overrides=[]
 for i,a in enumerate(argv):
  if a=="--pose-q":
   qidx=int(argv[i+1]);value=float(argv[i+2]);q[qidx]=value;overrides.append((qidx,value))
 if not overrides:raise RuntimeError("native-007 invocation has no scalar pose overrides")
 nheqPath=getarg("--joint-equality-payload")
 eqbytes,eqh,q,eqcorr=parse_nheq(nheqPath,q)
 pose,roots,loops=forward_fk(joints,sourcePose,q)
 nativePoseRecord=None
 if args.native_stderr is not None:
  for line in args.native_stderr.read_text().splitlines():
   prefix="registered_ligament_actual_pose_json="
   if line.startswith(prefix):
    nativePoseRecord=json.loads(line[len(prefix):]);break
 if nativePoseRecord is not None:
  if nativePoseRecord.get("schema")!="registered_ligament_actual_pose_v1":
   raise RuntimeError("native pose witness schema is unsupported")
  if nativePoseRecord.get("bonePayloadSha256")!=hashlib.sha256(ag.BONES.read_bytes()).hexdigest():
   raise RuntimeError("native pose witness bone SHA does not match exact invocation payload")
  native_q=nativePoseRecord.get("poseQ")
  if native_q is not None and (len(native_q)!=len(q) or max(abs(float(a)-float(b)) for a,b in zip(native_q,q))>2e-6):
   raise RuntimeError("native pose witness q differs from exact NHEQ-projected invocation q")
  native_poses=nativePoseRecord.get("bodyPoses")
  if not isinstance(native_poses,list) or len(native_poses)<max(max(j[:2]) for j in joints)+1:
   raise RuntimeError("native pose witness bodyPoses is shorter than the source articulated model")
  source_map=nativePoseRecord.get("sourceBodyMap")
  if not isinstance(source_map,list) or any(not isinstance(x,int) or x<0 or x>=len(native_poses) for x in source_map):
   raise RuntimeError("native pose witness sourceBodyMap is missing or out of bounds")
  pose=[np.asarray(p,dtype=np.float64) for p in native_poses]
 # Native root pose is a common SE(3) action; all projected points, normals and tests are frame-local.
 fk_default,_,_=forward_fk(joints,sourcePose,dq)
 fkErr=[]
 for i,p in enumerate(fk_default):
  if p is None or i not in sourcePose:continue
  dp=np.linalg.norm(p[:3]-np.asarray(sourcePose[i][:3]))
  dqerr=min(np.linalg.norm(p[3:]-sourcePose[i][3:]),np.linalg.norm(p[3:]+sourcePose[i][3:]))
  fkErr.append((dp,dqerr,i))
 maxP=max(v[0] for v in fkErr);maxQ=max(v[1] for v in fkErr)
 if maxP>1e-6 or maxQ>1e-6:raise RuntimeError(f"default FK does not reproduce source rest poses: {max(fkErr)}")
 currentPose={i:p for i,p in enumerate(pose) if p is not None}
 header_text,witness,names=base.parse_header();header_hash=hashlib.sha256(header_text.encode()).hexdigest()
 source={"catalogHeaderSha256":header_hash,"invocationPath":str(INV),"invocationSha256":hashlib.sha256(invb).hexdigest(),"nativeBinarySha256":inv.get("binary_sha256"),"sourceInvocationSha256":inv.get("source_sha256"),"rigidPath":argv[1],"rigidSha256":hashlib.sha256(rb).hexdigest(),"bonesPath":argv[3],"bonesSha256":hashlib.sha256(bb).hexdigest(),"nheqPath":str(nheqPath),"nheqSha256":hashlib.sha256(eqbytes).hexdigest(),"nativePoseQOverrides":overrides,"nheqMaximumCorrection":eqcorr,"rootPoseOverride":argv[argv.index("--root-pose")+1:argv.index("--root-pose")+8] if "--root-pose" in argv else None,"rootPoseTreatment":"common rigid transform omitted because all audit quantities are invariant in each source joint frame","boneAbi":bh[1],"boneRecordCount":bh[2],"vertexCount":bh[3],"indexCount":bh[4],"sourcePoseRecords":len(sourcePose),"sourceMissingBoneBodies":sorted({r["body"] for r in records.values()}-set(sourcePose)),"fkDefaultMaxPositionErrorM":maxP,"fkDefaultMaxQuaternionError":maxQ,"poseStateSource":"native stderr bodyPoses JSON" if nativePoseRecord is not None else "independent source-joint FK replay","nativePoseWitnessSchema":nativePoseRecord.get("schema") if nativePoseRecord is not None else None}
 out={"audit":"actual_pose_registered_ligament_catalog_source_geometry_v1","classification":"NHEQ-projected exact-invocation pose audit of inferred regional anchors; no subject-specific enthesis validation","source":source,"criteria":{"endpoint":"project hashed-header per-side frame seeds onto exact pinned source surface at commanded pose","normal":"transformed source vertex normals barycentrically interpolated at projected point; non-inward cosine >=-1e-4","segment":"scale-relative Moller-Trumbore with AABB culling against all 185 registered source meshes; interior t in (1e-5,1-1e-5)","anchor":"body-local float32 conversion followed by reconstruction in current pose","meshTopology":"open surface triangle clearance does not prove closed-volume containment","fk":"Metal operator scalar joint transform convention replayed in double precision; default state cross-checked against NHRIGID source poses"},"rows":[]}
 meshes_cache={};norm_cache={};allTried=[]
 for name in sorted(names):
  for side in (0,1):
   s0,s1=stable_pair(name,side);pair=frame_info(name,side);ji,frame=body_frame(joints,pose,pair)
   fk_joint_expected=(144 if side==0 else 130) if name in {"iliofemoral_lateral","ischiofemoral","iliofemoral_medial","pubofemoral"} else ((90 if side==0 else 40) if ("glenohumeral" in name or "coracohumeral" in name) else ((91 if side==0 else 41) if name=="elbow_rcl" else (150 if side==0 else 136)))
   if ji!=fk_joint_expected:raise RuntimeError(f"frame joint mismatch {name} side {side}: got {ji} expected {fk_joint_expected}")
   key=(pair,ji)
   if key not in meshes_cache:
    allm=lp.all_meshes(frame,bb,currentPose,records,vertex0,index0)
    if len(allm)!=len(records):raise RuntimeError(f"transformed {len(allm)}/{len(records)} meshes")
    meshes_cache[key]=allm
   allm=meshes_cache[key]
   if (s0,key) not in norm_cache:norm_cache[(s0,key)]=base.load_vertex_normals(s0,frame,bb,currentPose,records,vertex0)
   if (s1,key) not in norm_cache:norm_cache[(s1,key)]=base.load_vertex_normals(s1,frame,bb,currentPose,records,vertex0)
   seed0=np.asarray(witness[name][side][0],float);seed1=np.asarray(witness[name][side][1],float)
   p0,d0,t0=ag.project(tuple(seed0),allm[s0]);p1,d1,t1=ag.project(tuple(seed1),allm[s1])
   body0=records[s0]["body"];body1=records[s1]["body"]
   # Reproduce current frame-world point -> body local float32 GPU storage -> source frame point.
   def frame_to_bodylocal(pt,b):
    world=frame["origin"]+qrot(frame["q"],pt);bp=currentPose[b]
    return qrot([-bp[3],-bp[4],-bp[5],bp[6]],world-bp[:3])
   def bodylocal_to_frame(loc,b):
    bp=currentPose[b];world=bp[:3]+qrot(bp[3:],loc)
    return qrot([-frame["q"][0],-frame["q"][1],-frame["q"][2],frame["q"][3]],world-frame["origin"])
   loc0=np.asarray(frame_to_bodylocal(p0,body0),dtype=np.float32).astype(np.float64);loc1=np.asarray(frame_to_bodylocal(p1,body1),dtype=np.float32).astype(np.float64)
   a0=bodylocal_to_frame(loc0,body0);a1=bodylocal_to_frame(loc1,body1)
   n0=base.interpolated_normal(allm[s0],norm_cache[(s0,key)],t0,p0);n1=base.interpolated_normal(allm[s1],norm_cache[(s1,key)],t1,p1)
   delta=a1-a0;length=float(np.linalg.norm(delta));axis=delta/length if length>0 else np.zeros(3)
   cos0=float(np.dot(n0,axis));cos1=float(np.dot(n1,-axis))
   seglo=np.minimum(a0,a1)-1e-9;seghi=np.maximum(a0,a1)+1e-9;near=[];hits=[]
   for sid,(vv,tt) in allm.items():
    if np.all(np.max(vv,axis=0)>=seglo) and np.all(np.min(vv,axis=0)<=seghi):
     near.append(sid);hh=lp.vector_hits(a0,a1,lp.mesh_tuple(vv,tt))
     if hh:hits.append({"stableId":sid,"sourceRecord":records[sid]["sourceRecord"],"body":records[sid]["body"],"hits":[{"t":round(t,8),"triangle":ti} for t,ti in hh]})
   f0=ag.project(tuple(a0),allm[s0])[1];f1=ag.project(tuple(a1),allm[s1])[1]
   row={"name":name,"side":"L" if side==0 else "R","stableIds":[s0,s1],"sourceRecords":[records[s0]["sourceRecord"],records[s1]["sourceRecord"]],"endpointBodies":[body0,body1],"frameJointIndex":ji,"frameJointPair":list(pair),"headerSeedFrameM":[list(map(float,seed0)),list(map(float,seed1))],"projectedFrameMm":[[float(x*1000) for x in p0],[float(x*1000) for x in p1]],"seedProjectionResidualMm":[float(d0*1000),float(d1*1000)],"sourceProjectionTriangles":[t0,t1],"gpuLocalAnchorFloat32M":[loc0.tolist(),loc1.tolist()],"float32ReconstructedFrameMm":[[float(x*1000) for x in a0],[float(x*1000) for x in a1]],"float32AnchorSurfaceResidualMm":[float(f0*1000),float(f1*1000)],"referenceLengthMm":length*1000,"outwardNormalCos":[cos0,cos1],"allRegisteredBoneMeshesTested":len(allm),"aabbNearbyStableIds":near,"segmentTriangleHits":hits,"endpointMeshTopology":[base.topology(allm[s0][1]),base.topology(allm[s1][1])]}
   row["admittedByAudit"]=bool(max(d0,d1)<=.02 and min(cos0,cos1)>=-1e-4 and not hits and max(f0,f1)<=1e-5)
   out["rows"].append(row)
 if args.native_stderr is not None:
  native_lines=args.native_stderr.read_text().splitlines()
  admission_re=re.compile(r"registered_ligament_source_geometry=admitted\s+rows=(\d+)\s+retained_source_bone_meshes=(\d+)\s+source_bone_records=(\d+)\s+bone_payload_sha256=([0-9a-fA-F]{64})")
  admission_match=next((m for line in native_lines if (m:=admission_re.search(line))),None)
  binding={"invocationPath":str(INV),"invocationSha256":source["invocationSha256"],"nativeBinarySha256":source["nativeBinarySha256"],"sourceInvocationSha256":source["sourceInvocationSha256"],"catalogHeaderPath":str(HEADER),"catalogHeaderSha256":header_hash,"bonesPath":str(ag.BONES),"bonesSha256":source["bonesSha256"]}
  if admission_match:
   nr,nm,nb=int(admission_match.group(1)),int(admission_match.group(2)),int(admission_match.group(3));nsha=admission_match.group(4).lower()
   out["nativeSourceGeometryAdmission"]={"available":True,"admitted":nr==len(out["rows"]) and nm==len(records) and nb==len(records) and nsha==source["bonesSha256"],"stderrPath":str(args.native_stderr),"nativeRows":nr,"catalogRows":len(out["rows"]),"retainedSourceBoneMeshes":nm,"sourceBoneRecords":nb,"expectedSourceBoneRecords":len(records),"nativeBonePayloadSha256":nsha,"expectedBonePayloadSha256":source["bonesSha256"],"runBinding":binding,"scope":"native success marker is paired with exact invocation/binary/source hashes, current catalog-header hash, and reported bone hash; the native line itself does not carry the catalog-header hash"}
  else:
   out["nativeSourceGeometryAdmission"]={"available":False,"admitted":None,"stderrPath":str(args.native_stderr),"reason":"no successful registered_ligament_source_geometry=admitted record in stderr","runBinding":binding}
  diagnostic_re=re.compile(r"name=([^ ]+) side=([01]) facing=([-+0-9.eE]+),([-+0-9.eE]+) triangles=([0-9]+),([0-9]+).*?residuals_mm=([-+0-9.eE]+),([-+0-9.eE]+)")
  comparisons=[]
  for line in native_lines:
   m=diagnostic_re.search(line)
   if not m:continue
   name=m.group(1);side=int(m.group(2));r=next((x for x in out["rows"] if x["name"]==name and x["side"]==("L" if side==0 else "R")),None)
   if r is None:continue
   nc=[float(m.group(3)),float(m.group(4))];nt=[int(m.group(5)),int(m.group(6))];nr=[float(m.group(7)),float(m.group(8))]
   comparisons.append({"name":name,"side":r["side"],"nativeFacing":nc,"auditFacing":r["outwardNormalCos"],"facingAbsDelta":[abs(nc[i]-r["outwardNormalCos"][i]) for i in (0,1)],"nativeTriangles":nt,"auditTriangles":r["sourceProjectionTriangles"],"nativeResidualMm":nr,"auditResidualMm":r["seedProjectionResidualMm"],"residualAbsDeltaMm":[abs(nr[i]-r["seedProjectionResidualMm"][i]) for i in (0,1)],"matchesWithinTolerance":nt==r["sourceProjectionTriangles"] and max(abs(nc[i]-r["outwardNormalCos"][i]) for i in (0,1))<=3e-5 and max(abs(nr[i]-r["seedProjectionResidualMm"][i]) for i in (0,1))<=3e-5})
  out["nativeRuntimeDiagnosticComparison"]={"stderrPath":str(args.native_stderr),"available":bool(comparisons),"comparedRows":comparisons,"allComparedRowsMatch":all(x["matchesWithinTolerance"] for x in comparisons) if comparisons else None,"reason":None if comparisons else "stderr has no per-row cook-failure diagnostics; see nativeSourceGeometryAdmission for full-catalog success marker"}
 if hashlib.sha256(HEADER.read_bytes()).hexdigest()!=header_hash:raise RuntimeError("header changed during actual-pose audit")
 out["sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 path=args.output.resolve() if args.output is not None else ROOT/"tools/audit_actual_pose_registered_ligament_catalog.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+"\n")
 print("receipt",path,"sha256",out["sha256"],"rows",len(out["rows"]),"admitted",sum(x["admittedByAudit"] for x in out["rows"]))
 for r in out["rows"]:print(r["name"],r["side"],r["admittedByAudit"],"tri",r["sourceProjectionTriangles"],"res",list(round(x,6) for x in r["seedProjectionResidualMm"]),"cos",list(round(x,6) for x in r["outwardNormalCos"]),"hits",len(r["segmentTriangleHits"]))
 failures=[r for r in out["rows"] if not r["admittedByAudit"]]
 native_admission=out.get("nativeSourceGeometryAdmission")
 if failures:
  print("FAIL: rejected catalog rows",[(r["name"],r["side"]) for r in failures],file=sys.stderr)
  raise SystemExit(1)
 if native_admission is not None and native_admission.get("available") and not native_admission.get("admitted"):
  print("FAIL: native full-catalog source-geometry admission did not match this header/bones/counts",file=sys.stderr)
  raise SystemExit(1)
if __name__=="__main__":main()
