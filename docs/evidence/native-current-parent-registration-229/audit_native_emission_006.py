from pathlib import Path
from fractions import Fraction
import csv,json,hashlib,re,ast,importlib.util,sys,time,signal
import numpy as np
RUN=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/P/native-kind-aware-source-preflight-226/run-006-native-emission")
CAN=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/volume-worker/skin-source-033/prepared-007-kind-aware-canonical-compact")
OUT=RUN/"skin-audit-001"
OUT.mkdir(exist_ok=True)
NCSV=RUN/"registered-skin-100-native.csv.candidate033.csv"
NCAN=CAN/"registered-O.csv"
FCAN=CAN/"registered-O-faces.csv"
PR=json.loads((CAN/"receipt.json").read_text())
SRCPOSE=Path(PR["inputs"]["source-poses"]["path"])
NPOSE=RUN/"registered-skin-poses-100.json"
STDOUT=RUN/"stdout.txt"
NRECEIPT=RUN/"receipt.json"
def sha(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open(newline="") as f:
  return list(csv.DictReader(f))
def qmat(q):
 x,y,z,w=np.asarray(q,dtype=np.float64)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],np.float64)
# Parse the common placement from the native log.
line=next(x for x in STDOUT.read_text().splitlines() if "candidate033_native_source_frame_measurement" in x)
match=re.search(r"common_placement_translation_m=(\[[^]]+\]).*common_placement_rotation=(\[[^]]+\])",line)
if not match: raise ValueError("native stdout lacks common placement transform")
t=np.asarray(ast.literal_eval(match.group(1)),np.float64)
R=np.asarray(ast.literal_eval(match.group(2)),np.float64).reshape(3,3)
# Load compact canonical and actual native emitted rows.
nr=rows(NCSV); cr=rows(NCAN); fr=rows(FCAN)
if len(nr)!=32002 or len(cr)!=32002 or len(fr)!=63959:
 raise ValueError(f"native/canonical row count mismatch: {len(nr)}/{len(cr)}/{len(fr)}")
for i,(a,b) in enumerate(zip(nr,cr)):
 if int(a["vertex"])!=i or int(b["vertex"])!=i: raise ValueError(f"point row identity gap {i}")
for i,a in enumerate(fr):
 if int(a["face"])!=i: raise ValueError(f"face row identity gap {i}")
Ntxt=np.asarray([[float(r[k]) for k in ("wx","wy","wz")] for r in nr],np.float64)
Ctxt=np.asarray([[float(r[k]) for k in ("wx","wy","wz")] for r in cr],np.float64)
N=Ntxt.astype(np.float32); C=Ctxt.astype(np.float32)
F=np.asarray([[int(r[k]) for k in ("v0","v1","v2")] for r in fr],np.int64)
Nd=np.asarray([int(r["donor"]) for r in nr],np.int64)
Cd=np.asarray([int(r["donor"]) for r in cr],np.int64)
Nn=np.asarray([[float(r[k]) for k in ("nx","ny","nz")] for r in nr],np.float64)
Cn=np.asarray([[float(r[k]) for k in ("nx","ny","nz")] for r in cr],np.float64)
Nl=np.asarray([[float(r[k]) for k in ("lx","ly","lz")] for r in nr],np.float64)
Nm=np.asarray([[float(r[f"m{a}{b}"]) for a in range(3) for b in range(3)] for r in nr],np.float64).reshape(-1,3,3)
# Common-frame point, face, and normal checks.
Rerr=float(np.max(np.abs(R.T@R-np.eye(3))))
Rdet=float(np.linalg.det(R))
Expected64=C.astype(np.float64)@R.T+t
Expected32=Expected64.astype(np.float32)
Delta=N.astype(np.float64)-Expected64
Dnorm=np.linalg.norm(Delta,axis=1)
ExactF32=np.all(N.view(np.uint32)==Expected32.view(np.uint32),axis=1)
def ordered_f32_bits(x):
 b=x.view(np.uint32).astype(np.uint64)
 return np.where((b&0x80000000)!=0,(~b)&0xffffffff,b|0x80000000)
Ulp=np.abs(ordered_f32_bits(N).astype(np.int64)-ordered_f32_bits(Expected32).astype(np.int64))
valid=bool(F.ndim==2 and F.shape[1]==3 and F.size and F.min()>=0 and F.max()<len(N))
repeat=int(np.sum((F[:,0]==F[:,1])|(F[:,1]==F[:,2])|(F[:,2]==F[:,0]))) if valid else -1
if valid:
 NT=N[F].astype(np.float64); ET=Expected64[F]
 NCross=np.cross(NT[:,1]-NT[:,0],NT[:,2]-NT[:,0])
 ECross=np.cross(ET[:,1]-ET[:,0],ET[:,2]-ET[:,0])
 NArea=np.linalg.norm(NCross,axis=1); EArea=np.linalg.norm(ECross,axis=1)
 Dot=np.einsum("ij,ij->i",NCross,ECross)
 Cos=Dot/np.maximum(NArea*EArea,1e-300)
 Deg=int(np.sum((NArea<=0)|(EArea<=0)))
 Flip=int(np.sum((Dot<=0)&(NArea>0)&(EArea>0)))
else:
 NArea=EArea=Dot=Cos=np.array([]);Deg=Flip=-1
# Registered current normals and donor-local rows against actual native body poses.
NPOSED=json.loads(NPOSE.read_text()); SPOSED=json.loads(SRCPOSE.read_text())
if (NPOSED.get("schema"),NPOSED.get("registration_fingerprint32"),NPOSED.get("binding_count")) != (SPOSED.get("schema"),SPOSED.get("registration_fingerprint32"),SPOSED.get("binding_count")):
 raise ValueError("native pose schema/fingerprint/binding count differs from source")
SB={int(x["body_index"]):x for x in SPOSED["bodies"]}
NB={int(x["body_index"]):x for x in NPOSED["bodies"]}
if set(SB)!=set(NB) or len(SB)!=86: raise ValueError("native/source pose body sets differ")
curp=[];curr=[];restp=[];restr=[]
for body in sorted(SB):
 s=SB[body];n=NB[body]
 p0=np.asarray(s["current"]["position_world_m"],float);p1=np.asarray(n["current"]["position_world_m"],float)
 curp.append((float(np.linalg.norm(p1-(R@p0+t))),body))
 A=qmat(s["current"]["orientation_world_xyzw"]);B=qmat(n["current"]["orientation_world_xyzw"])
 curr.append((float(np.max(np.abs(B-R@A))),body))
 restp.append((float(np.linalg.norm(np.asarray(n["rest"]["position_world_m"],float)-np.asarray(s["rest"]["position_world_m"],float))),body))
 restr.append((float(np.max(np.abs(qmat(n["rest"]["orientation_world_xyzw"])-qmat(s["rest"]["orientation_world_xyzw"])))),body))
POSEM={b:qmat(NB[b]["current"]["orientation_world_xyzw"]) for b in NB}
POSEP={b:np.asarray(NB[b]["current"]["position_world_m"],float) for b in NB}
FrameErr=[];LocalErr=[]
for i,b in enumerate(Nd):
 if int(b) not in POSEM: continue
 FrameErr.append((float(np.max(np.abs(Nm[i]-POSEM[int(b)].T))),i))
 LocalErr.append((float(np.linalg.norm(Nm[i]@(N.astype(float)[i]-POSEP[int(b)])-Nl[i])),i))
NnLen=np.linalg.norm(Nn,axis=1)
NnUnit=Nn/np.maximum(NnLen[:,None],1e-300)
CnUnit=Cn/np.maximum(np.linalg.norm(Cn,axis=1)[:,None],1e-300)
EnUnit=CnUnit@R.T
NormalCos=np.einsum("ij,ij->i",NnUnit,EnUnit)
NormalErr=np.linalg.norm(NnUnit-EnUnit,axis=1)
bad=np.flatnonzero((NArea<=0)|(EArea<=0)|(Dot<=0)) if valid else np.array([],dtype=int)
witness=[]
for i in bad[:64]:
 witness.append({"face_row":int(i),"vertex_rows":F[i].tolist(),"dot":float(Dot[i]),"native_area2":float(NArea[i]),"expected_area2":float(EArea[i])})
maxcp=max(curp,key=lambda x:x[0]);maxcr=max(curr,key=lambda x:x[0])
maxrp=max(restp,key=lambda x:x[0]);maxrr=max(restr,key=lambda x:x[0])
maxfr=max(FrameErr,key=lambda x:x[0]);maxlr=max(LocalErr,key=lambda x:x[0])
cheap={
 "status":"CHEAP_GATES_COMPLETE",
 "classification":"actual native emitted geometry and pose parity audit only; no physical qualification",
 "native_csv":str(NCSV),"native_csv_sha256":sha(NCSV),
 "native_pose_sha256":sha(NPOSE),"source_pose_sha256":sha(SRCPOSE),
 "canonical_nodes_sha256":sha(NCAN),"canonical_faces_sha256":sha(FCAN),
 "canonical_preparation_receipt_sha256":sha(CAN/"receipt.json"),
 "native_stdout_sha256":sha(STDOUT),"native_emission_receipt":json.loads(NRECEIPT.read_text()),
 "common_placement":{"translation_m":t.tolist(),"rotation_row_major":R.ravel().tolist(),"determinant":Rdet,"orthogonality_max_abs_error":Rerr},
 "counts":{"native_nodes":len(N),"faces":len(F),"native_f32_exact_rows":int(np.all(Ntxt==N.astype(float),axis=1).sum()),
 "native_donors_match_canonical":int(np.sum(Nd==Cd)),"repeated_index_faces":repeat,"native_or_expected_degenerate_faces":Deg,"flipped_orientation_faces":Flip},
 "point_parity":{"max_axis_residual_m":float(np.max(np.abs(Delta),initial=0)),"max_euclidean_residual_m":float(Dnorm.max(initial=0)),
 "p95_euclidean_residual_m":float(np.quantile(Dnorm,.95)),"exact_expected_f32_rows":int(ExactF32.sum()),
 "max_float32_ulp_delta":int(Ulp.max(initial=0)),"worst_vertex_row":int(np.argmax(Dnorm))},
 "face_orientation":{"min_cosine":float(Cos.min(initial=1)),"min_native_area2_m2":float(NArea.min(initial=np.inf)),
 "bad_face_witnesses":witness},
 "normal_parity":{"min_cosine":float(NormalCos.min(initial=1)),"max_vector_residual":float(NormalErr.max(initial=0))},
 "pose_parity":{"body_count":len(SB),"max_current_position_residual_m":maxcp[0],"max_current_position_body":maxcp[1],
 "max_current_rotation_matrix_residual":maxcr[0],"max_current_rotation_body":maxcr[1],
 "max_rest_position_residual_m":maxrp[0],"max_rest_position_body":maxrp[1],
 "max_rest_rotation_matrix_residual":maxrr[0],"max_rest_rotation_body":maxrr[1]},
 "native_frame_reconstruction":{"max_world_to_local_matrix_residual":maxfr[0],"worst_matrix_vertex":maxfr[1],
 "max_local_reconstruction_residual_m":maxlr[0],"worst_local_vertex":maxlr[1]},
 "source_rest_orientation_from_canonical_preparation":{"flipped_faces":PR["checks"]["source_rest_unposed_orientation_audit"]["flipped_or_degenerate_faces"],
 "degenerate_faces":PR["checks"]["source_rest_unposed_orientation_audit"]["degenerate_faces"],
 "minimum_face_cosine":PR["checks"]["source_rest_unposed_orientation_audit"]["minimum_face_cosine"]},
 "gates":{"all_points_finite":bool(np.isfinite(N).all()),"face_indices_valid":valid,
 "no_repeated_or_degenerate_faces":bool(repeat==0 and Deg==0),"donors_match_canonical":bool(np.array_equal(Nd,Cd)),
 "point_common_frame_residual_under_2um":bool(Dnorm.max(initial=0)<=2e-6),"no_face_orientation_flips":bool(Flip==0),
 "current_body_pose_parity_under_2um_and_1e-6_rotation":bool(maxcp[0]<=2e-6 and maxcr[0]<=1e-6),
 "source_rest_pose_parity_under_1e-6":bool(maxrp[0]<=1e-6 and maxrr[0]<=1e-6),
 "native_donor_frame_reconstruction_under_2um":bool(maxlr[0]<=2e-6 and maxfr[0]<=1e-6)}
}
(OUT/"cheap-gates.json").write_text(json.dumps(cheap,indent=2,sort_keys=True)+"\n")
print(json.dumps({"cheap_gates":cheap["gates"],"counts":cheap["counts"],"point_parity":cheap["point_parity"],
 "face_orientation":{k:v for k,v in cheap["face_orientation"].items() if k!="bad_face_witnesses"},
 "normal_parity":cheap["normal_parity"],"pose_parity":cheap["pose_parity"],
 "native_frame_reconstruction":cheap["native_frame_reconstruction"],"report":str(OUT/"cheap-gates.json")},indent=2))
# Skip exact predicates only for unusable coordinates/index/degenerate rows.
if not np.isfinite(N).all() or not valid or repeat or Deg:
 (OUT/"self.json").write_text(json.dumps({"status":"SKIPPED_INVALID_INPUT","complete":False,"physical_admission":False},indent=2)+"\n")
 sys.exit(2)
# Reuse the previously pinned exact rational same-surface predicate.
tool=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/boolean-prototype-001/audit_self_envelope006.py")
spec=importlib.util.spec_from_file_location("native006_exact_owner",tool)
audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit)
pred=audit.load_exact();t0=time.monotonic()
def on_alarm(signum,frame): raise TimeoutError("exact self audit exceeded 180 seconds")
signal.signal(signal.SIGALRM,on_alarm);signal.alarm(180)
try:
 vertices=tuple(tuple(Fraction.from_float(float(x)) for x in point) for point in N)
 records=pred._records(vertices,tuple(tuple(int(v) for v in face) for face in F))
 result=pred._audit_pair(records,records,same_surface=True)
 pairs=[]
 for a,b in result["triangle_pairs"][:256]:
  a=int(a);b=int(b);pts=pred.triangle_intersection_points(records[a][0],records[b][0])
  pairs.append({"face_rows":[a,b],"vertex_rows":[F[a].tolist(),F[b].tolist()],
   "native_triangle_f32":[N[F[a]].tolist(),N[F[b]].tolist()],
   "intersection_points_fraction":[[str(c) for c in p] for p in pts]})
 selfreport={"status":"COMPLETE_RATIONAL_SAME_SURFACE_AUDIT","candidate_csv":str(NCSV),"candidate_csv_sha256":sha(NCSV),
 "canonical_face_table_sha256":sha(FCAN),"point_count":int(len(N)),"triangle_count":int(len(F)),
 "exact_predicate_owner":str(audit.EXACT_OWNER),"exact_predicate_sha256":sha(audit.EXACT_OWNER),
 "exact_triangle_pair_count":int(result["count"]),"aabb_candidate_pairs":int(result["aabb_candidate_pairs"]),
 "allowed_shared_vertex_or_edge_pairs":int(result["allowed_shared_vertex_or_edge_pairs"]),
 "witnesses_retained":len(pairs),"witnesses_truncated":int(result["count"])>len(pairs),"triangle_pairs":pairs,
 "common_placement":cheap["common_placement"],"complete":True,"physical_steps":0,"physical_admission":False,
 "elapsed_s":time.monotonic()-t0}
 (OUT/"self.json").write_text(json.dumps(selfreport,indent=2,sort_keys=True)+"\n")
 print(json.dumps({k:v for k,v in selfreport.items() if k!="triangle_pairs"},indent=2))
finally: signal.alarm(0)
