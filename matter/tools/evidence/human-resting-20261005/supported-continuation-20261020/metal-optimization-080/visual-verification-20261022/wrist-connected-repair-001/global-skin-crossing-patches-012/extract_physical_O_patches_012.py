#!/usr/bin/env python3
"""Extract source-indexed, edge-connected P1 physical-O crossing patch inputs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.spatial import cKDTree

R=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=R/"global-skin-crossing-patches-012"
O=R/"global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz"
P=R/"global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz"
C=R/"global-inner-p1-full-domain-clipping-009/exact-O-pair-011/run-001/canonical-pair-classification.npz"
Z=R/"global-inner-p1-full-domain-clipping-009/validation-001/zero-capacity-witness-index-009.npz"
INV=R/"registered-surrounding-tissue-native013-191/inventory.json"
SKIN=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
POSE=R/"wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json"
HELPER=R/"registered-pose-frame-helper-194/tools"
N_RINGS=2
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()

def adjacency(faces):
 adj=[set() for _ in range(len(faces))];edge_owners={}
 for fi,t in enumerate(np.asarray(faces)):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
   e=(int(min(a,b)),int(max(a,b)));edge_owners.setdefault(e,[]).append(fi)
 for owners in edge_owners.values():
  if len(owners)>1:
   root=owners[0]
   for v in owners[1:]:adj[root].add(v);adj[v].add(root)
 return [sorted(x) for x in adj]

def expand(seed,adj,nrings):
 level=np.full(len(adj),-1,np.int16);q=list(map(int,np.unique(seed)));level[q]=0
 for u in q:
  if level[u]>=nrings:continue
  for v in adj[u]:
   if level[v]<0:level[v]=level[u]+1;q.append(v)
 inc=set(np.flatnonzero(level>=0).tolist());seen=set();comps=[]
 for s in sorted(inc):
  if s in seen:continue
  st=[s];seen.add(s);cc=[]
  while st:
   u=st.pop();cc.append(u)
   for v in adj[u]:
    if v in inc and v not in seen:seen.add(v);st.append(v)
  comps.append(np.asarray(sorted(cc),np.int64))
 return level,comps

def closest(P,A,B,C):
 ab=B-A;ac=C-A;ap=P-A
 d00=np.einsum("ij,ij->i",ab,ab);d01=np.einsum("ij,ij->i",ab,ac);d11=np.einsum("ij,ij->i",ac,ac)
 d20=np.einsum("ij,ij->i",ap,ab);d21=np.einsum("ij,ij->i",ap,ac)
 den=d00*d11-d01*d01;safe=np.maximum(den,np.finfo(float).tiny)
 v=(d11*d20-d01*d21)/safe;w=(d00*d21-d01*d20)/safe
 inside=(den>1e-28)&(v>=0)&(w>=0)&(v+w<=1)
 bd=np.full(len(P),np.inf);bb=np.zeros((len(P),3))
 for U,W,edge in ((A,B,0),(B,C,1),(C,A,2)):
  e=W-U;t=np.clip(np.einsum("ij,ij->i",P-U,e)/np.maximum(np.einsum("ij,ij->i",e,e),np.finfo(float).tiny),0,1)
  q=U+t[:,None]*e;d=np.einsum("ij,ij->i",P-q,P-q);m=d<bd;bd[m]=d[m]
  bary=np.stack((1-t,t,np.zeros_like(t)),1) if edge==0 else (np.stack((np.zeros_like(t),1-t,t),1) if edge==1 else np.stack((t,np.zeros_like(t),1-t),1))
  bb[m]=bary[m]
 proj=A+v[:,None]*ab+w[:,None]*ac;pd=np.einsum("ij,ij->i",P-proj,P-proj);m=inside&(pd<bd)
 bd[m]=pd[m];bb[m]=np.stack((1-v-w,v,w),1)[m]
 return np.sqrt(bd),bb

def map_surface(Q,X,F):
 tri=np.asarray(X,dtype=np.float64)[np.asarray(F,dtype=np.int64)]
 ct=cKDTree(tri.mean(1));vt=cKDTree(X)
 vd,vi=vt.query(Q,k=1);outf=np.full(len(Q),-1,np.int64);outb=np.zeros((len(Q),3));dist=np.full(len(Q),np.inf);stable=np.zeros(len(Q),bool)
 for lo in range(0,len(Q),256):
  hi=min(lo+256,len(Q)); ids=ct.query(Q[lo:hi],k=min(48,len(F)))[1]; ids=np.asarray(ids)
  if ids.ndim==1:ids=ids[:,None]
  ids128=ct.query(Q[lo:hi],k=min(128,len(F)))[1];ids128=np.asarray(ids128)
  if ids128.ndim==1:ids128=ids128[:,None]
  for j,(ii,jj) in enumerate(zip(ids,ids128)):
   p=Q[lo+j];t=tri[ii];d,b=closest(np.broadcast_to(p,(len(ii),3)),t[:,0],t[:,1],t[:,2]);k=int(np.argmin(d))
   outf[lo+j]=int(ii[k]);outb[lo+j]=b[k];dist[lo+j]=d[k]
   tw=tri[jj];dw,_=closest(np.broadcast_to(p,(len(jj),3)),tw[:,0],tw[:,1],tw[:,2]);stable[lo+j]=dw.min()<=d[k]+1e-10
 code=np.zeros(len(Q),np.uint8);code[vd<=2e-7]=1;code[(vd>2e-7)&(dist<=2e-6)&stable]=2
 return vi.astype(np.int64),vd,outf,outb,dist,stable,code

def barycentric(point,tri):
 a,b,c=np.asarray(tri,float);p=np.asarray(point,float);ab=b-a;ac=c-a;ap=p-a
 d00=ab@ab;d01=ab@ac;d11=ac@ac;d20=ap@ab;d21=ap@ac;den=d00*d11-d01*d01
 if den<=1e-30:raise ValueError("degenerate O face")
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;b=np.array([1-v-w,v,w]);q=b@np.asarray(tri,float)
 return b,float(np.linalg.norm(q-p))

def run():
 D.mkdir(parents=True,exist_ok=True)
 for f in (O,P,C,Z,INV,SKIN,POSE,HELPER/"rebind_human_skin_reference.py"):
  if not f.is_file():raise FileNotFoundError(f)
 o=np.load(O,allow_pickle=False);p=np.load(P,allow_pickle=False);c=np.load(C,allow_pickle=False);z=np.load(Z,allow_pickle=False)
 if any(not np.array_equal(c[k],p[k]) for k in ("bank_ids","bank_local_face_ids","physical_O_face_ids")):raise ValueError("exact011 rows mismatch exact pair input")
 x=o["points_world_m"];of=o["physical_outer_faces"].astype(np.int64);meta=json.loads(INV.read_text())["banks"]
 if x.dtype!=np.float32 or len(meta)!=689:raise ValueError("wrong O mesh or typed bank inventory")
 b=c["bank_ids"].astype(int);lf=c["bank_local_face_ids"].astype(int);fid=c["physical_O_face_ids"].astype(int)
 z9={}
 for i,(bi,li) in enumerate(zip(z["key_bank_ids"],z["key_local_face_ids"])):
  a,h=map(int,z["key_witness_row_offsets"][i:i+2]);z9[(int(bi),int(li))]=z9.get((int(bi),int(li)),False) or bool(np.any(z["zero_capacity_mask"][a:h]))
 from9=np.asarray([z9.get((int(bi),int(li)),False) for bi,li in zip(b,lf)])
 adj=adjacency(of);level,comps=expand(fid,adj,N_RINGS);patchface=np.full(len(of),-1,np.int32)
 for j,fc in enumerate(comps):patchface[fc]=j
 pairpatch=patchface[fid]
 if np.any(pairpatch<0):raise AssertionError("unowned seed pair")
 # Preserve identity and source CSR for every selected node.
 faceoff=[0];nodeoff=[0];seedoff=[0];edgeoff=[0];pairoff=[0]
 fids=[];fnodes=[];flocal=[];fring=[];nids=[];nxyz=[];seedids=[];edges=[];prows=[]
 srcptr=[0];srcids=[];srcw=[]
 for pid,fc in enumerate(comps):
  nd=np.unique(of[fc].ravel());lookup={int(v):i for i,v in enumerate(nd)}
  local=np.asarray([[lookup[int(v)] for v in tri] for tri in of[fc]],np.int64)
  # Selected patch boundary: one selected incidence per undirected edge.
  ee=np.concatenate((of[fc][:,[0,1]],of[fc][:,[1,2]],of[fc][:,[2,0]]));ee.sort(axis=1);ee=np.unique(ee,axis=0)
  counts={}
  for tri in of[fc]:
   for aa,bb in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
    e=(int(min(aa,bb)),int(max(aa,bb)));counts[e]=counts.get(e,0)+1
  be=np.asarray([e for e,n in counts.items() if n==1],np.int64).reshape(-1,2) if counts else np.empty((0,2),np.int64)
  sf=np.intersect1d(fc,np.unique(fid));pr=np.flatnonzero(pairpatch==pid)
  fids.extend(fc.tolist());fnodes.extend(of[fc].tolist());flocal.extend(local.tolist());fring.extend(level[fc].tolist())
  nids.extend(nd.tolist());nxyz.extend(x[nd].tolist());seedids.extend(sf.tolist());edges.extend(be.tolist());prows.extend(pr.tolist())
  faceoff.append(len(fids));nodeoff.append(len(nids));seedoff.append(len(seedids));edgeoff.append(len(edges));pairoff.append(len(prows))
  for node in nd:
   a,h=map(int,o["node_source_indptr"][node:node+2]);srcids.extend(o["node_source_indices"][a:h].tolist());srcw.extend(o["node_source_weights"][a:h].tolist());srcptr.append(len(srcids))
 # Exact hit-point barycentric support on original O triangle.
 hitp=p["exact_hit_offsets"];hp=p["exact_hit_points_m"]; hb=[];hr=[];hface=[];hitoff=[0]
 pairrows=np.asarray(prows,np.int64)
 for r in pairrows:
  oi=int(fid[r]);tri=x[of[oi]].astype(np.float64)
  a,h=map(int,hitp[r:r+2])
  for q in hp[a:h]:
   bary,res=barycentric(q,tri)
   if res>2e-6 or bary.min()<-2e-5 or bary.max()>1+2e-5:raise ValueError(f"exact hit barycentric invalid pair {r}: {res}, {bary}")
   hb.append(bary);hr.append(res);hface.append(oi)
  hitoff.append(len(hb))
 # Map selected physical-O nodes to source294's registered current surface.
 sys.path.insert(0,str(HELPER));import rebind_human_skin_reference as rb
 payload=rb.decode(SKIN.read_bytes());pose=json.loads(POSE.read_text())
 sx=rb.registered_positions_float32(payload,pose,frame="current");sf=payload["inds"]
 q=np.asarray(nxyz,dtype=np.float64);vi,vd,mf,mb,md,stable,method=map_surface(q,sx,sf)
 # Typed-bank body ownership CSR for each source pair.
 pairbanks=b[pairrows];bodyptr=[0];body=[]
 for bi in pairbanks:body.extend(meta[int(bi)]["body_indices"]);bodyptr.append(len(body))
 out=D/"connected-physical-O-patches-012.npz"
 np.savez_compressed(out,schema=np.asarray("numi.human.global-skin-crossing-patches.v1"),
  patch_face_offsets=np.asarray(faceoff,np.int64),patch_face_ids=np.asarray(fids,np.int64),patch_face_node_ids=np.asarray(fnodes,np.int64),
  patch_face_local_node_ids=np.asarray(flocal,np.int64),patch_face_ring_level=np.asarray(fring,np.int16),
  patch_node_offsets=np.asarray(nodeoff,np.int64),patch_node_ids=np.asarray(nids,np.int64),patch_node_points_world_m=np.asarray(nxyz,np.float32),
  patch_node_source_indptr=np.asarray(srcptr,np.int64),patch_node_source_indices=np.asarray(srcids,np.int64),patch_node_source_weights=np.asarray(srcw,np.float64),
  patch_seed_face_offsets=np.asarray(seedoff,np.int64),patch_seed_face_ids=np.asarray(seedids,np.int64),
  patch_boundary_edge_offsets=np.asarray(edgeoff,np.int64),patch_boundary_edge_node_ids=np.asarray(edges,np.int64),
  patch_pair_offsets=np.asarray(pairoff,np.int64),pair_row_ids=pairrows,pair_bank_ids=pairbanks.astype(np.uint16),
  pair_bank_stable_ids=np.asarray([meta[int(i)]["stable_id"] for i in pairbanks],np.uint32),
  pair_bank_layers=np.asarray([meta[int(i)]["layer"] for i in pairbanks],np.uint8),
  pair_bank_kind_codes=np.asarray([0 if meta[int(i)]["kind"]=="soft_tissue" else 1 for i in pairbanks],np.uint8),
  pair_bank_body_indptr=np.asarray(bodyptr,np.int64),pair_bank_body_indices=np.asarray(body,np.int32),
  pair_bank_local_face_ids=lf[pairrows].astype(np.uint32),pair_O_face_ids=fid[pairrows].astype(np.uint32),
  pair_class_codes=c["class_codes"][pairrows].astype(np.uint8),pair_from_original009_zero_support=from9[pairrows],
  pair_witness_tet_ids=p["witness_tet_ids"][pairrows].astype(np.uint32),
  pair_exact_hit_offsets=np.asarray(hitoff,np.int64),
  pair_exact_hit_points_m=np.concatenate([hp[int(hitp[r]):int(hitp[r+1])] for r in pairrows],axis=0),
  pair_hit_O_face_ids=np.asarray(hface,np.uint32),pair_hit_O_barycentric=np.asarray(hb,float).reshape(-1,3),
  pair_hit_O_projection_residual_m=np.asarray(hr,float),
  source294_mapping_direct_vertex_ids=vi,source294_mapping_direct_vertex_distance_m=vd,
  source294_mapping_face_ids=mf,source294_mapping_barycentric=mb,source294_mapping_face_residual_m=md,
  source294_mapping_stable_48_vs_128=stable,source294_mapping_method_codes=method)
 summaries=[]
 for pid,fc in enumerate(comps):
  pr=np.flatnonzero(pairpatch==pid);ubs=np.unique(b[pr])
  summaries.append({"patch_id":pid,"face_count":len(fc),"node_count":len(np.unique(of[fc])),"seed_face_count":len(np.intersect1d(fc,np.unique(fid))),
   "pair_count":len(pr),"pairs_from_original009_zero_support":int(from9[pr].sum()),
   "bank_ids":ubs.astype(int).tolist(),"stable_ids":sorted(set(int(meta[int(k)]["stable_id"]) for k in ubs)),
   "ring_level_counts":{str(int(k)):int(v) for k,v in zip(*np.unique(level[fc],return_counts=True))},
   "boundary_edge_count":int(edgeoff[pid+1]-edgeoff[pid]),"face_id_min":int(fc.min()),"face_id_max":int(fc.max())})
 inp=[O,P,C,Z,INV,SKIN,POSE,HELPER/"rebind_human_skin_reference.py"]
 m={"source_payload_sha256":sha(SKIN),"source_vertices":payload["nv"],"source_faces":len(sf),"source_fingerprint32":payload["fingerprint"],
  "pose_snapshot_sha256":sha(POSE),"frame":"current",
  "direct_vertex_candidates":int(np.sum(method==1)),"barycentric_face_candidates":int(np.sum(method==2)),
  "unmatched":int(np.sum(method==0)),"max_residual_m":float(md.max()),"p50_residual_m":float(np.median(md)),
  "p95_residual_m":float(np.quantile(md,.95)),"max_mapped_residual_m":float(md[method>0].max()) if np.any(method>0) else None,
  "stable_48_vs_128_count":int(stable.sum()),"global_nearest_certified":False,
  "method_codes":{"0":"unmatched","1":"nearest source294 vertex within 0.2um","2":"nearest source294 triangle within 2um and 48-vs-128 centroid candidates stable"},
  "note":"Barycentric correspondence is a bounded nearest-face candidate map, not certified global closest point or topology identity."}
 rep={"schema":"numi.human.global-skin-crossing-patches.report.v1","status":"PATCH_INPUTS_EXTRACTED_NO_REPAIR_OR_ADMISSION",
  "native_admission":False,"physical_admission":False,"rings_expanded":N_RINGS,"inputs_sha256":{str(f):sha(f) for f in inp},
  "source_O_mesh":{"path":str(O),"sha256":sha(O),"nodes":len(x),"physical_outer_nodes":len(o["physical_outer_node_ids"]),"physical_outer_faces":len(of)},
  "exact011":{"path":str(C),"sha256":sha(C),"pairs":len(c["bank_ids"]),"class_code_counts":{str(int(k)):int(v) for k,v in zip(*np.unique(c["class_codes"],return_counts=True))},
   "exact011_report_sha256":sha(C.parent/"report.json"),"pair_seed_input_sha256":sha(P)},
  "original009_zero_support":{"path":str(Z),"sha256":sha(Z),"zero_rows":int(z["zero_capacity_mask"].sum()),
   "unique_zero_bank_face_keys":int(sum(z9.values())),"canonical011_pairs_linked":int(from9.sum()),
   "canonical011_pairs_not_in_009_partial_prefix":int((~from9).sum())},
  "selection":{"seed_faces":int(len(np.unique(fid))),"connected_patch_count":len(comps),"selected_faces":len(fids),"selected_nodes":len(nids),
   "selected_pairs":len(pairrows),"boundary_edges":len(edges),"duplicate_face_rows":len(fids)-len(np.unique(fids)),"patches":summaries},
  "contact_constraints":{"exact_hit_points":len(hb),"max_O_triangle_projection_residual_m":float(max(hr,default=0)),
   "barycentric_min":float(np.min(hb)),"barycentric_max":float(np.max(hb)),
   "meaning":"exact011 pair hits with barycentric support on the unchanged O face; repair input only, not solved force constraints."},
  "source294_current_pose_mapping":m,
  "ancestry":{"O_face":"patch_face_ids indexes original 008 physical_outer_faces rows",
   "O_node":"patch_node_ids indexes original 008 points_world_m rows",
   "O_source_CSR":"patch_node_source_* copies original 008 source interpolation per O node",
   "detailed294":"source294_mapping_* maps each O node into source294 current-pose vertices/faces"},
  "limitations":["009 is a processed prefix; unlinked does not imply absence of zero support or physical crossing.",
   "Source294 map is candidate-bounded and not certified global nearest or topology identity.",
   "No repair, force solve, pose sweep, full physical qualification, or native admission was performed."]}
 rp=D/"patch-report-012.json";rp.write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
 return rp,out,rep

if __name__=="__main__":
 rp,out,r=run();m=r["source294_current_pose_mapping"]
 print(json.dumps({"report":str(rp),"arrays":str(out),"patches":r["selection"]["connected_patch_count"],
  "selected_faces":r["selection"]["selected_faces"],"selected_nodes":r["selection"]["selected_nodes"],
  "selected_pairs":r["selection"]["selected_pairs"],"source294_direct":m["direct_vertex_candidates"],
  "source294_barycentric":m["barycentric_face_candidates"],"source294_unmatched":m["unmatched"],
  "source294_max_residual_m":m["max_residual_m"]},indent=2))

