from __future__ import annotations
import csv, gzip, json, hashlib, importlib.util, runpy, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
D=Path("/Users/n/Work/numi-human-connected-surface-069")
OUT=D/"connected-reference-capacity-112/fine-source-probe-002"
OUT.mkdir(parents=True,exist_ok=False)
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
RUN=D/"connected-reference-capacity-112/run-002"
BASE=D/"connected-cell-witness-103/run-001/baseline"
COMPACT=D/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
TRANSFER=D/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz"
DETAIL=E/"bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"
ROWS=E/"accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv"
REGCSV=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/registered-skin.csv"
BANK=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
AUDIT=Path("/Users/n/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
CAP=D/"connected-reference-capacity-112/human_skin_fat_reference_capacity.py"
PRED=D/"connected-cell-trust-107/matter/root_exact_segment_witness.py"
RAY=D/"connected-cell-trust-107/matter/tools/human_skin_fat_bone_clearance.py"
TARGET=[204,215,217,218,219,231,233,271,337,1467,1990,2024,2187,2261,2262]
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def pin(path,expected):
 got=sha(path)
 if got!=expected: raise ValueError(f"SHA mismatch {path}: {got}")
 return got
pins={
 "candidate":pin(BASE/"candidate.npz","ddb86fbf07183adce0194c59a1acfc5456c96cb2efc709eb587e5489d3d59163"),
 "compact084":pin(COMPACT,"5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db"),
 "Wnew":sha(TRANSFER),"nhskin":pin(DETAIL,"7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb"),
 "accepted_transfer_rows":pin(ROWS,"71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14"),
 "registered_skin_csv":pin(REGCSV,"eccbc683f8450f27c8c2559471aff8fa8c62644871011d46fe3c3dbfdf3f2524"),
 "query_pairs":pin(BASE/"query/pairs.csv","95023ef49a865473ab8baaca65da37bc45d78d5b4adf9e16ee7effa604bad4eb"),
 "query_nodes":pin(BASE/"query/nodes.csv","e49ef0e61ed9551bc3782e3bb8def6a45b5b2b32124ce1c7b7dabb9e286badef"),
 "query_points":pin(BASE/"query/points.csv","f15ea806a61c6a70fb4d051c932df22ac173b1f39e3f60662fad396362a8b58d"),
 "run112_report":sha(RUN/"report.json"),"run112_audits":sha(RUN/"exact-segment-audits.jsonl.gz")}
with open(ROWS,newline="") as f: rows=list(csv.DictReader(f))
detail_xyz=np.array([[float(r[f"detail_rest_world_{a}_m"]) for a in "xyz"] for r in rows],dtype=np.float64)
active=np.array([int(r["active"]) for r in rows],dtype=bool)
if len(rows)!=61950 or detail_xyz.shape!=(61950,3):raise ValueError("detail-row source shape mismatch")
# exact 039/075 source frame match
reg=list(csv.DictReader(REGCSV.open(newline="")))
if len(reg)!=len(rows):raise ValueError("registered/detail row count mismatch")
regxyz=np.array([[float(r[f"w{a}"]) for a in "xyz"] for r in reg],dtype=np.float64)
if not np.array_equal(detail_xyz,regxyz): raise ValueError("accepted source rows differ from pinned registered 039 vertices")
with np.load(TRANSFER,allow_pickle=False) as z:
 W=csr_matrix((z["data"],z["indices"],z["indptr"]),shape=tuple(map(int,z["shape"])))
if W.shape!=(61950,7360):raise ValueError(f"unexpected W shape {W.shape}")
with np.load(COMPACT,allow_pickle=False) as z:
 R=np.asarray(z["reference_points_m"],dtype=np.float64)
 C=np.asarray(z["current037_points_m_f32"],dtype=np.float32)
with np.load(BASE/"candidate.npz",allow_pickle=False) as z:
 P=np.asarray(z["points_world_f32"],dtype=np.float32)
if not np.array_equal(P.view(np.uint32),C.view(np.uint32)):raise ValueError("baseline/compact current bits mismatch")
# Current detailed surface induced by the exact registered W map, not an independent registration.
detail_current=detail_xyz+W.dot(C.astype(np.float64)-R)
# Parse NHSKIN topology without changing payload bytes.
raw=DETAIL.read_bytes()
import struct
magic,abi,nb,nv,ni,_=struct.unpack_from("<8s5I",raw)
vo=60+nb*36;io=vo+nv*56
faces=np.frombuffer(raw,dtype="<u4",count=ni,offset=io).reshape(-1,3).astype(np.int64,copy=True)
if len(faces)!=108987:raise ValueError(f"unexpected detailed face count {len(faces)}")
# vectorized exact closest point-on-triangle search, chunks avoid a large temporary.
def closest_on_mesh(p, xyz, F, chunk=12000):
 best=(float("inf"),None,None)
 for lo in range(0,len(F),chunk):
  ff=F[lo:lo+chunk]; T=xyz[ff]; A=T[:,0]; B=T[:,1]; Cc=T[:,2]
  v0=B-A;v1=Cc-A;w=p-A
  d00=np.einsum("ij,ij->i",v0,v0);d01=np.einsum("ij,ij->i",v0,v1);d11=np.einsum("ij,ij->i",v1,v1)
  d20=np.einsum("ij,ij->i",w,v0);d21=np.einsum("ij,ij->i",w,v1);den=d00*d11-d01*d01
  u=np.full(len(ff),np.nan);v=np.full(len(ff),np.nan);good=den>1e-30
  u[good]=(d11[good]*d20[good]-d01[good]*d21[good])/den[good]
  v[good]=(d00[good]*d21[good]-d01[good]*d20[good])/den[good]
  proj=A+u[:,None]*v0+v[:,None]*v1
  inside=good&(u>=0)&(v>=0)&(u+v<=1)
  bestp=np.zeros_like(A);bestb=np.zeros((len(ff),3));bestd=np.full(len(ff),np.inf)
  if np.any(inside):
   bestp[inside]=proj[inside];bestb[inside]=np.column_stack((1-u[inside]-v[inside],u[inside],v[inside]))
   bestd[inside]=np.einsum("ij,ij->i",proj[inside]-p,proj[inside]-p)
  for e,(i,j) in enumerate(((0,1),(1,2),(2,0))):
   X=T[:,i];Y=T[:,j];d=Y-X;dd=np.einsum("ij,ij->i",d,d)
   t=np.zeros(len(ff));ok=dd>1e-30;t[ok]=np.clip(np.einsum("ij,ij->i",p-X[ok],d[ok])/dd[ok],0,1)
   q=X+t[:,None]*d;dist=np.einsum("ij,ij->i",q-p,q-p);take=dist<bestd
   if np.any(take):
    bestp[take]=q[take];bestd[take]=dist[take];bb=np.zeros((np.count_nonzero(take),3));bb[:,i]=1-t[take];bb[:,j]=t[take];bestb[take]=bb
  k=int(np.argmin(bestd));d2=float(bestd[k])
  if d2<best[0]:best=(d2,lo+k,(bestp[k],bestb[k]))
 return best
# typed bank reader & exact classifier
sys.path.insert(0,str(AUDIT.parent))
import human_skin_fat_bone_audit as audit
tri,ids,gids,bodies,matrices,offsets,positions,bankkeys,bankpins,exact=audit.read_bank_inputs(BANK,D/"connected-cell-trust-107/matter")
sys.path.insert(0,str(CAP.parent));import human_skin_fat_reference_capacity as cap
ray=runpy.run_path(str(RAY)); exact_parameter=ray["_exact_t_barycentric"]; classifier=exact["classify_segment_triangle"]
# Endpoint classifications from complete 185-bank endpoint query (missing pair means AABB proved outside).
with open(BASE/"query/pairs.csv",newline="") as f: pairrows=list(csv.DictReader(f))
pairmap={(int(r["query_row"]),int(r["bank_body"]),int(r["bank_stable_id"])):r for r in pairrows}
with open(BASE/"query/nodes.csv",newline="") as f: node_rows=list(csv.DictReader(f))
inside_by_row=defaultdict(set)
for (qr,b,bo),r in pairmap.items():
 if r["classification"]=="inside":inside_by_row[qr].add((b,bo))
# Read exact segment banks that had events, not full 185-bank requery.
route_banks=defaultdict(set)
with gzip.open(RUN/"exact-segment-audits.jsonl.gz","rt") as f:
 for line in f:
  r=json.loads(line); c=int(r["column_row"])
  if c not in TARGET:continue
  res=r.get("result",{})
  if res.get("proper_hit_count",0)>0 or res.get("unresolved_witnesses") or r.get("status")!="PASS":route_banks[c].add((int(r["body"]),int(r["bone"])))
# Build complete typed banks lazily for culprit set.
prepared={}
def getbank(key):
 if key in prepared:return prepared[key]
 body,bone=key; m=(ids[:,0]==body)&(ids[:,1]==bone); bt=tri[m]; bid=ids[m,3].astype(np.int64)
 rec={"triangles_local":bt,"triangle_stable_ids":bid,"expected_triangle_count":len(bt),"aabb_min_local":bt.min(axis=(0,1)),"aabb_max_local":bt.max(axis=(0,1)),"world_to_local":matrices[body],"body_position_world":positions[body],"com_offset":offsets[body]}
 prepared[key]=cap.prepare_bone_bank(rec);return prepared[key]
def source_endpoint_status(O,Q,start_status,bank):
 if np.array_equal(np.asarray(O,dtype=np.float64),np.asarray(Q,dtype=np.float64)):
  return start_status,{"proper_hits":[],"unresolved":[],"status":"same_point_as_O"}
 triB=bank.triangles; a=cap.world_to_bank_local(O,body_position_world=bank.body_position_world,world_to_local=bank.world_to_local,com_offset=bank.com_offset); b=cap.world_to_bank_local(Q,body_position_world=bank.body_position_world,world_to_local=bank.world_to_local,com_offset=bank.com_offset)
 lo=np.minimum(a,b);hi=np.maximum(a,b);mask=np.all(bank.triangle_hi>=lo,axis=1)&np.all(bank.triangle_lo<=hi,axis=1)
 hits=[];unres=[]
 for ix in np.flatnonzero(mask):
  ix=int(ix);r=classifier(a,b,triB[ix]);kind=r.get("classification")
  if kind in ("disjoint","parallel_disjoint"):continue
  if kind!="proper_interior_hit":unres.append({"triangle_stable_id":int(bank.stable_ids[ix]),"classification":kind});continue
  t,bary=exact_parameter(a,b,triB[ix]);hits.append((t,int(bank.stable_ids[ix]),float(t),[float(v) for v in bary]))
 hits.sort(key=lambda h:(h[0],h[1]))
 ties=[]
 for l,r in zip(hits,hits[1:]):
  if l[0]==r[0]:ties.append([l[1],r[1],str(l[0])])
 end=start_status if len(hits)%2==0 else ("outside" if start_status=="inside" else "inside")
 state="REJECT" if unres or ties else "PARITY_INFERRED"
 return end,{"status":state,"start_status":start_status,"parity_inferred_end_status":end,"proper_hits":[{"t_num":str(t.numerator),"t_den":str(t.denominator),"t_float":tf,"triangle_stable_id":sid,"barycentric":bary} for t,sid,tf,bary in hits],"unresolved":unres,"equal_parameter_ties":ties,"candidate_triangle_count":int(np.count_nonzero(mask)),"scope":"exact segment hit parity from pinned classifier, conditional on exact start class; no standalone ray classification/watertightness proof"}
# current endpoint source topology & donor data
rowsout=[];started=time.time()
for c in TARGET:
 O=P[c].astype(np.float64)
 d2,faceidx,closest=closest_on_mesh(O,detail_current,faces)
 Q,bary=closest
 verts=faces[faceidx].astype(int)
 # Active W support and typed transfer donor records for the source face vertices.
 support=[]
 for v in verts:
  lo,hi=W.indptr[v:v+2];cols=W.indices[lo:hi];vals=W.data[lo:hi]
  support.append({"detail_row":int(v),"active":bool(active[v]),"detail_donor_body":int(rows[v]["detail_donor_body"]),"transfer_donor_body":int(rows[v]["transfer_donor_body"]),"source_face_row":int(rows[v]["selected_face_row"]),"coarse_support_rows":cols.astype(int).tolist(),"weights":[float(x) for x in vals]})
 # triangulated current source outward area normal; winding is retained as authored and no outward sign is claimed
 A,B,Cc=detail_current[verts];cr=np.cross(B-A,Cc-A);area=float(np.linalg.norm(cr)*.5);n=cr/np.linalg.norm(cr) if np.linalg.norm(cr)>1e-30 else np.zeros(3)
 endps={layer:inside_by_row[c+layer*2411] for layer in range(3)}
 candidates=set().union(*endps.values(),route_banks[c])
 bank_reports=[]
 for key in sorted(candidates):
  bank=getbank(key); Ostatus="inside" if key in endps[0] else "outside"
  qstatus,seg=source_endpoint_status(O,Q,Ostatus,bank)
  bank_reports.append({"body":key[0],"stable_bone_id":key[1],"O_endpoint_status":Ostatus,"I_endpoint_status":"inside" if key in endps[1] else "outside","D_endpoint_status":"inside" if key in endps[2] else "outside","fine_nearest_anchor_status":qstatus,"O_to_anchor_exact_segment":seg,"complete_bank_triangle_count":bank.expected_triangle_count})
 rowsout.append({"column_row":c,"O_world_current037_m":O.tolist(),"nearest_registered_detailed_surface_point_world_current037_m":Q.tolist(),"distance_O_to_detail_surface_m":float(np.sqrt(d2),),"detail_source_face_row":int(faceidx),"detail_vertex_rows":verts.tolist(),"barycentric_on_detail_face":bary.tolist(),"detail_triangle_area_m2":area,"authored_winding_unit_normal_diagnostic_only":n.tolist(),"detail_face_transfer_support":support,"endpoint_inside_banks":{"O":[list(x) for x in sorted(endps[0])],"I":[list(x) for x in sorted(endps[1])],"D":[list(x) for x in sorted(endps[2])]},"route_event_banks":[list(x) for x in sorted(route_banks[c])],"bank_comparisons":bank_reports})
report={"schema":"fine-source-vs-bone-15column-probe-v1","status":"READ_ONLY_DIAGNOSTIC","input_pins":pins,"scope":{"columns":TARGET,"candidate_current_points":"Exact Float32 103/084 baseline points; detailed surface induced in current037 frame by Wnew*(C-R) from pinned 075 accepted-transfer rows.","surface_distance":"Vectorized Euclidean closest point on actual candidate.nhskin triangles; source face and barycentrics retained.","bone_test":"Exact proper-hit parity from known 103 O point class along O-to-nearest-detail-anchor segment, only for culprit banks seen inside O/I/D or as exact segment-event banks in frozen112. Anchor endpoint class is inferred by parity, not independent all-bank ray query.","limitations":["This does not query all185 banks for each nearest anchor.","Endpoint parity classification is conditional on pinned known O status and exact segment having no unresolved/tied contacts; it is not an independent point ray certificate.","Registered detailed current points are induced by Wnew P1 displacement; not a new native surface emission.","No geometry changed and no admission claim."]},"elapsed_seconds":time.time()-started,"detail_face_count":int(len(faces)),"results":rowsout}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"out":str(OUT/"report.json"),"elapsed":report["elapsed_seconds"],"cols":len(rowsout),"pins":pins},indent=2))
