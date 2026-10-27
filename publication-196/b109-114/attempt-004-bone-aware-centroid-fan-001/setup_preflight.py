#!/usr/bin/env python3
"""Lightweight topology/ancestry/witness-map setup for 109/114 fan constructor."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib,json
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
WORK=S/"per-bank-workers-20261027/b109-114"
PATCHDIR=WORK/"attempt-001-boundary-feasibility/patch-components"
GRAPH=WORK/"attempt-002-exact-contours/exact-contour-graph.json"
OUT=WORK/"attempt-004-bone-aware-centroid-fan-001"
C=ROOT/"D/publication-182-stage-001/combined-653/candidate.npz"
A=C.parent/"audit-159/report.json"
BASE_SHA="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
AUDIT_SHA="c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e"
GRAPH_SHA="524e1c62a7768b5ec9ae50a2b7bc5ac65bd2303ff9d03355486bf784cfc932ab"
PATCH_SHA=["331294b1fa5cd727015db683181daebd80599b5f7217a0eef9b1b18a4ed3b84d","55af4b716aeacdcc8a25fb951f0b7fc84ad24908d996c5c9641ab08ac439c661"]
FILES=[PATCHDIR/"component-00-face-rows.npy",PATCHDIR/"component-01-face-rows.npy"]
SOURCE=OUT/"fan-source-patch.npz";WITNESS=OUT/"actual-crossing-witness-map.json";REPORT=OUT/"setup-preflight.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def boundary_rows(rows,F):
 uses=Counter()
 for fi in rows:
  a,b,c=map(int,F[int(fi)])
  for u,v in ((a,b),(b,c),(c,a)):uses[tuple(sorted((u,v)))]+=1
 if any(n not in (1,2) for n in uses.values()):raise RuntimeError("nonmanifold patch edge incidence")
 return sorted(e for e,n in uses.items() if n==1)
def bary2(point,tri):
 x,y=point[1],point[2];(x0,y0),(x1,y1),(x2,y2)=[(t[1],t[2]) for t in tri]
 den=(x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
 if not den:return None
 w0=((x1-x)*(y2-y)-(x2-x)*(y1-y))/den
 w1=((x2-x)*(y0-y)-(x0-x)*(y2-y))/den
 return (w0,w1,1-w0-w1)
def map_child(lam,mu):
 e0=(Fraction(1),Fraction(0),Fraction(0));e1=(Fraction(0),Fraction(1),Fraction(0));e2=(Fraction(0),Fraction(0),Fraction(1))
 for i,t in enumerate(((e0,e1,mu),(e1,e2,mu),(e2,e0,mu))):
  w=bary2(lam,t)
  if w is not None and min(w)>=0 and sum(w)==1:return i,w
 return None
def metrics(P,F,parentP,parentF,parents):
 tri=P[F].astype(np.float64);par=parentP[parentF[parents]].astype(np.float64)
 pn=np.cross(par[:,1]-par[:,0],par[:,2]-par[:,0]);cn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 den=np.einsum("ij,ij->i",pn,pn);ori=np.einsum("ij,ij->i",pn,cn)/den;ratio=np.linalg.norm(cn,axis=1)/np.sqrt(den)
 angles=[];aspect=[]
 for t in tri:
  lengths=[np.linalg.norm(t[(i+1)%3]-t[i]) for i in range(3)]
  area=np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))/2
  ang=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
   ang.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-300),-1,1)))))
  angles.append(min(ang));aspect.append(max(lengths)**2/(4*np.sqrt(3)*max(area,1e-300)))
 return {"child_face_count":len(F),"min_parent_area_ratio":float(ratio.min()),"min_signed_orientation_ratio":float(ori.min()),
  "orientation_all_positive":bool(np.all(ori>0)),"minimum_child_angle_deg":float(min(angles)),
  "maximum_aspect_measure":float(max(aspect))}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if sha(C)!=BASE_SHA or sha(A)!=AUDIT_SHA or sha(GRAPH)!=GRAPH_SHA:raise RuntimeError("pinned source inputs changed")
 for i,p in enumerate(FILES):
  if sha(p)!=PATCH_SHA[i]:raise RuntimeError(f"parent-face component {i} pin changed")
 with np.load(C,allow_pickle=False) as z:
  P=z["points_world_m"].astype(np.float32);F=z["source_O_faces"].astype(np.int64)
  si=z["face_source_ancestry_indptr"];sr=z["face_source_ancestry_face_rows"]
  ui=z["face_fine_subface_ancestry_indptr"];ur=z["face_fine_subface_ancestry_rows"]
  di=z["face_fine_detail_ancestry_indptr"];dr=z["face_fine_detail_ancestry_rows"]
 G=json.loads(GRAPH.read_text());A0=json.loads(A.read_text())
 bank=next(x for x in A0["banks"] if (int(x["body"]),int(x["stable_bone_id"]))==(109,114))
 pairs={tuple(map(int,x)) for x in bank["result"]["triangle_pairs"]}
 comps=[np.load(p,allow_pickle=False).astype(np.int64) for p in FILES]
 if [len(x) for x in comps]!=[239,38]:raise RuntimeError("expected distinct 239/38 parent patches")
 bounds=[boundary_rows(x,F) for x in comps]
 if [len(x) for x in bounds]!=[49,14]:raise RuntimeError(f"fixed boundary edge counts changed: {[len(x) for x in bounds]}")
 verts=[set(map(int,np.unique(F[x].reshape(-1)))) for x in comps]
 if verts[0]&verts[1]:raise RuntimeError("separate source patches share vertex identities")
 for fi in np.concatenate(comps):
  if any(len(ar[int(fi)+1:int(fi)+2])<0 for ar in (si,ui,di)):raise RuntimeError("invalid ancestry pointer")
  if not (si[fi+1]>si[fi] and ui[fi+1]>ui[fi] and di[fi+1]>di[fi]):raise RuntimeError(f"source face {int(fi)} missing ancestry")
 parent=np.concatenate(comps);sourcev=np.unique(F[parent].reshape(-1));local={int(v):i for i,v in enumerate(sourcev)}
 per_face_mu={};center=[]
 for fi in parent:
  t=P[F[int(fi)]].astype(np.float64)
  l=np.asarray([np.linalg.norm(t[2]-t[1]),np.linalg.norm(t[0]-t[2]),np.linalg.norm(t[1]-t[0])])
  mu=.5*l/l.sum()+np.ones(3)/6
  per_face_mu[int(fi)]=tuple(Fraction.from_float(float(x)) for x in mu)
  center.append(mu@t)
 points=np.vstack([P[sourcev].astype(np.float64),np.asarray(center)]).astype(np.float32)
 slot={int(fi):i for i,fi in enumerate(parent)};faces=[];parents=[];tags=[];bary=[]
 EYE=np.eye(3)
 for ci,rows in enumerate(comps):
  for fi0 in rows:
   fi=int(fi0);a,b,c=map(int,F[fi]);x,y,z=(local[a],local[b],local[c]);m=len(sourcev)+slot[fi];mu=np.asarray([float(v) for v in per_face_mu[fi]])
   faces.extend(([x,y,m],[y,z,m],[z,x,m]));parents.extend([fi]*3);tags.extend([ci]*3)
   bary.extend((np.asarray([EYE[0],EYE[1],mu]),np.asarray([EYE[1],EYE[2],mu]),np.asarray([EYE[2],EYE[0],mu])))
 boundarylocal=np.asarray([(local[u],local[v]) for rows in bounds for u,v in rows],np.int64)
 faces=np.asarray(faces,np.int64);parents=np.asarray(parents,np.int64);tags=np.asarray(tags,np.int8);bary=np.asarray(bary,np.float64)
 if len(boundarylocal)!=63 or len(np.unique(boundarylocal))!=63:raise RuntimeError("two fixed loops should have 63 unique edges/nodes")
 # Verify all exact saved contour segments and endpoint barycentric witnesses map to a face-fan child.
 recmap=defaultdict(list)
 for r in G["endpoint_records"]:recmap[(int(r["pair_index"]),int(r["root"]))].append(r)
 witnesses=[];covered=set()
 for ci,cg in enumerate(G["per_component_graphs"]):
  rows=set(map(int,comps[ci]))
  for edge in cg["contour_edge_roots"]:
   sf=int(edge["skin_face_row"]);bf=int(edge["bone_face_row"]);pi=int(edge["pair_index"])
   if (sf,bf) not in pairs or sf not in rows:raise RuntimeError(f"crossing pair {(sf,bf)} not covered by source patch")
   for root in map(int,edge["endpoint_roots"]):
    rr=[r for r in recmap[(pi,root)] if int(r["skin_face_row"])==sf]
    if len(rr)!=1:raise RuntimeError(f"pair/root {(pi,root)} has {len(rr)} source-face endpoint records")
    r=rr[0];lam=tuple(Fraction(x) for x in r["skin_barycentrics"])
    if sum(lam)!=1 or min(lam)<0:raise RuntimeError(f"invalid endpoint barycentrics for pair {pi}")
    mapped=map_child(lam,per_face_mu[sf])
    if mapped is None:raise RuntimeError(f"exact crossing endpoint from pair {pi} falls outside all fan children")
    k,w=mapped;child=3*slot[sf]+k
    witnesses.append({"component":ci,"pair_index":pi,"endpoint_record_id":int(r["id"]),"root":root,
     "skin_face_row":sf,"bone_face_row":bf,"fan_child_face_index":child,
     "skin_barycentrics":[str(x) for x in lam],"bone_barycentrics":r["bone_barycentrics"],
     "local_m":r["local_m"],"fan_child_weights":[str(x) for x in w]})
    covered.add((pi,root))
 if len(pairs)!=187 or len(witnesses)!=374 or len(covered)!=374:raise RuntimeError("expected all 187 pairs and 374 endpoint witnesses")
 np.savez_compressed(SOURCE,source_candidate_node_rows=np.r_[sourcev,np.full(len(parent),-1,np.int64)],
  patch_outer_boundary_edges=boundarylocal,original_patch_face_rows=parent,
  component_parent_face_rows_00=comps[0],component_parent_face_rows_01=comps[1],
  component_id=tags,fan_source_child_faces=faces,fan_child_parent_face_rows=parents,
  point_parent_candidate_face_rows=np.r_[np.full(len(sourcev),-1,np.int64),parent],
  point_parent_candidate_barycentric_f64=np.vstack([np.zeros((len(sourcev),3)),np.asarray([per_face_mu[int(fi)] for fi in parent],np.float64)]))
 WITNESS.write_text(json.dumps({"schema":"numi.human.b109_114.actual_crossing_fan_witnesses.v1",
  "candidate_sha256":BASE_SHA,"exact_graph_sha256":GRAPH_SHA,"witnesses":witnesses},indent=2,sort_keys=True)+"\n")
 q=metrics(points,faces,P,F,parents)
 report={"schema":"numi.human.b109_114.centroid_fan_setup_preflight.v1","status":"SETUP_PREFLIGHT_READY",
  "candidate_sha256":sha(C),"audit159_sha256":sha(A),"exact_graph_sha256":sha(GRAPH),
  "source_component_face_files":[{"path":str(p),"sha256":sha(p),"face_count":len(comps[i]),"fixed_boundary_edges":len(bounds[i])} for i,p in enumerate(FILES)],
  "fan_source_patch_path":str(SOURCE),"fan_source_patch_sha256":sha(SOURCE),
  "actual_crossing_witness_map_path":str(WITNESS),"actual_crossing_witness_map_sha256":sha(WITNESS),
  "two_components_remain_disconnected":True,"parent_face_count":len(parent),"fan_child_face_count":len(faces),
  "source_vertex_count":len(sourcev),"fan_center_count":len(parent),"fixed_boundary_edge_count":len(boundarylocal),
  "crossing_pair_count":len(pairs),"crossing_endpoint_constraint_count":len(witnesses),
  "witness_mapping":"374 rational endpoint records mapped to fan children by parent-face barycentric coordinates; stable bone face and pair identities retained.",
  "float32_fan_quality":q,"fit_or_geometry_audit_run":False,"geometry_admitted":False,"physical_steps":0}
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"report":str(REPORT),"report_sha256":sha(REPORT),"fan_source_patch":str(SOURCE),
  "fan_source_patch_sha256":sha(SOURCE),"witness_map":str(WITNESS),"witness_map_sha256":sha(WITNESS),
  "status":report["status"],"float32_fan_quality":q,"crossing_pair_count":len(pairs),"crossing_endpoint_constraints":len(witnesses)},indent=2))
if __name__=="__main__":main()

