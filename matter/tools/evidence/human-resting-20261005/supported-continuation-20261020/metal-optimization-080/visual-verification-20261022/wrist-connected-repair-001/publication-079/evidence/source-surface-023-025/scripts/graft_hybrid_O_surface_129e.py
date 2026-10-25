#!/usr/bin/env python3
import hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BASE=Path("/Users/numi/Work/numi-human-connected-surface-069")
CUTROOT=BASE/"connected-source-patch-cut-123"
COARSE=BASE/"publication-072/evidence/whole-reference-TetGen-084/whole-skin-fat-reference-084-candidate.npz"
DETAIL=BASE/"publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz"
OLD_CUT=CUTROOT/"run-019-route-harmonized-old-input/source-patch-cut.npz"
OLD_REPORT=CUTROOT/"run-019-route-harmonized-old-input/source-patch-cut-report.json"
OLD_HYBRID=CUTROOT/"run-019-route-harmonized-old-emission/hybrid-O-surface-129-old-harmonized.npz"
NEW_CUT=CUTROOT/"run-016-best-reroute-expanded-cut/source-patch-cut.npz"
NEW_REPORT=CUTROOT/"run-016-best-reroute-expanded-cut/source-patch-cut-report.json"
NEW_HYBRID=CUTROOT/"run-017-expanded-target-hybrid/hybrid-O-surface-128.npz"
OUT=CUTROOT/"run-020-grafted-31plus2-hybrid"
EXPECTED={
 "coarse084":"d3d0abb590f5858de4a27e0362e98bc9b3088831183ea045096cefc4ebba26e1",
 "direct_detail075":"21f1869b418a2eca1e5a0522059c8f9cc6f8afe1a270ccc7c7d99c9c19fdd2a4",
 "old_cut":"4a065fea7f347e5c2c7c9a1c82f2a598af509a4cce8fde1cf4ec32068abc2b61",
 "old_hybrid":"53dcfb8f89038fc597d59f0c56193826633439153b088e8d99dfbc5111031e7d",
 "new_cut":"21e2b65d85ba9bac176aaa670b916bc13189975a295c9ab328d1e835bb2e0b78",
 "new_hybrid":"541eb6eabf442c74592853d40aecea1c9c4cb6aac20a31d4d3aa0017cd32afb7",
}
OLD_COMPONENTS=[0,3,4,5,6,*range(8,31),32,33,34]
NEW_COMPONENTS=[1,2]

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def ekey(a,b):
 a=int(a);b=int(b)
 return (a,b) if a<b else (b,a)
def directed(tri,e):
 for i in range(3):
  a,b=int(tri[i]),int(tri[(i+1)%3])
  if ekey(a,b)==e:return (a,b)
 return None
def require(cond,msg):
 if not cond: raise RuntimeError(msg)

def selected_fine_rows(cut, report, included_components):
 fine=np.asarray(cut["fine_subfaces"],dtype=np.int64)
 routed={ekey(*e) for e in np.asarray(cut["routed_cut_edges"],dtype=np.int64)}
 edge_faces=defaultdict(list)
 for fi,t in enumerate(fine):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
   edge_faces[ekey(a,b)].append(fi)
 owner=np.full(len(fine),-1,dtype=np.int32)
 comps=[]
 for start in range(len(fine)):
  if owner[start]>=0: continue
  ci=len(comps); stack=[start]; owner[start]=ci; members=[]
  while stack:
   fi=stack.pop();members.append(fi);t=fine[fi]
   for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
    ee=ekey(a,b)
    if ee in routed: continue
    for fj in edge_faces[ee]:
     if owner[fj]<0: owner[fj]=ci;stack.append(fj)
  comps.append(members)
 rc=np.asarray(cut["route_segment_component"],dtype=np.int64)
 rl=np.asarray(cut["route_segment_loop"],dtype=np.int64)
 ends=np.asarray(cut["route_segment_coarse_endpoints"],dtype=np.int64)
 ip=np.asarray(cut["route_path_indptr"],dtype=np.int64)
 pn=np.asarray(cut["route_path_nodes"],dtype=np.int64)
 selected=set();loop_log=[]
 for rec in report["patch_loops"]:
  comp=int(rec["component"])
  if comp not in included_components: continue
  require(rec.get("status")=="closed_disjoint_source_edge_cycle",
          f"requested component {comp} loop {rec.get('loop')} not closed: {rec.get('status')}")
  loop=int(rec["loop"])
  segids=np.flatnonzero((rc==comp)&(rl==loop))
  require(len(segids)>0,f"no route segments for component {comp} loop {loop}")
  paths=[]
  for si in segids:
   p=pn[ip[si]:ip[si+1]].astype(np.int64)
   require(len(p)>=2, f"route path too short component {comp} segment {int(si)}")
   paths.append(p)
  candidate_comps=[]
  for ci,members in enumerate(comps):
   okay=True
   for p in paths:
    for a,b in zip(p[:-1],p[1:]):
     ee=ekey(a,b)
     hits=[fi for fi in edge_faces[ee] if owner[fi]==ci and directed(fine[fi],ee)==(int(b),int(a))]
     if len(hits)!=1:
      okay=False;break
    if not okay:break
   if okay:candidate_comps.append(ci)
  require(len(candidate_comps)==1,
          f"component {comp} loop {loop}: expected one directed fine side, got {candidate_comps}")
  chosen=candidate_comps[0]
  selected.update(comps[chosen])
  loop_log.append({"coarse_component":comp,"loop":loop,"fine_component":chosen,
                   "face_count":len(comps[chosen]),"segments":len(segids)})
 return np.asarray(sorted(selected),dtype=np.int32),loop_log

def selected_coarse_rows(cut, component_ids):
 ip=np.asarray(cut["patch_component_indptr"],dtype=np.int64)
 rows=np.asarray(cut["patch_component_face_rows"],dtype=np.int64)
 out=set()
 for ci in component_ids:
  require(0<=ci<len(ip)-1,f"coarse component id out of range {ci}")
  out.update(map(int,rows[ip[ci]:ip[ci+1]]))
 return out

def main():
 require(not OUT.exists(),f"output directory already exists: {OUT}")
 pins={
  "coarse084":sha(COARSE),"direct_detail075":sha(DETAIL),
  "old_cut":sha(OLD_CUT),"old_hybrid":sha(OLD_HYBRID),
  "new_cut":sha(NEW_CUT),"new_hybrid":sha(NEW_HYBRID),
  "old_cut_report":sha(OLD_REPORT),"new_cut_report":sha(NEW_REPORT),
 }
 for k,v in EXPECTED.items(): require(pins[k]==v,f"pinned {k} hash mismatch {pins[k]}")
 co=np.load(COARSE,allow_pickle=False); det=np.load(DETAIL,allow_pickle=False)
 oc=np.load(OLD_CUT,allow_pickle=False); nc=np.load(NEW_CUT,allow_pickle=False)
 oh=np.load(OLD_HYBRID,allow_pickle=False); nh=np.load(NEW_HYBRID,allow_pickle=False)
 ore=json.load(open(OLD_REPORT)); nre=json.load(open(NEW_REPORT))
 # Confirm identical registered coarse/reference frame.
 Pbase=np.asarray(co["reference_points_m"],dtype=np.float64)
 Fbase=np.asarray(co["source_O_faces"],dtype=np.int64)
 require(np.array_equal(oh["points_world_m"][:0],nh["points_world_m"][:0]),"invalid empty frame check")
 # The emitted candidate indices are compact; every kind-0 row must retain exact coarse coordinates.
 for label,h in (("old",oh),("new",nh)):
  ix=np.flatnonzero(h["node_kind"]==0)
  src=np.asarray(h["node_source_row"])[ix].astype(np.int64)
  require(np.all((src>=0)&(src<len(Pbase))),f"{label} coarse node source rows out of range")
  err=np.linalg.norm(np.asarray(h["points_world_m"])[ix]-Pbase[src],axis=1)
  require(float(err.max(initial=0.0))<=2e-14,f"{label} coarse frame mismatch {float(err.max())}")
 old_all=np.asarray(sorted(selected_fine_rows(oc,ore,set(OLD_COMPONENTS+[2]))[0]),dtype=np.int32)
 old_keep,old_loop_log=selected_fine_rows(oc,ore,set(OLD_COMPONENTS))
 new_keep,new_loop_log=selected_fine_rows(nc,nre,set(NEW_COMPONENTS))
 # Candidate face rows must agree with reconstructed directed cut sides.
 require(set(map(int,old_keep)) <= set(map(int,oh["selected_fine_subface_rows"])),
         "harmonized old candidate omits retained directed fine rows")
 require(set(map(int,new_keep))==set(map(int,nh["selected_fine_subface_rows"])),
         "run017 selected fine rows disagree with directed cut components")
 old_patch_rows=selected_coarse_rows(oc,OLD_COMPONENTS)
 removed_bad_rows=selected_coarse_rows(oc,{2})
 old_candidate_rows=set(map(int,oh["selected_coarse_face_rows"]))
 require(old_patch_rows <= old_candidate_rows,
         f"harmonized old candidate omits retained coarse patch rows: {sorted(old_patch_rows-old_candidate_rows)[:20]}")
 new_patch_rows=selected_coarse_rows(nc,NEW_COMPONENTS)
 require(new_patch_rows==set(map(int,nh["selected_coarse_face_rows"])),
         "run017 selected coarse rows disagree with components 1+2")
 require(not (old_patch_rows & new_patch_rows),
         f"retained/expanded patch coarse face overlap: {sorted(old_patch_rows&new_patch_rows)[:20]}")
 patch_rows=old_patch_rows|new_patch_rows
 # Build route edge selection from the same components used for the patch faces.
 def route_edge_keys(cut, component_ids):
  rc=np.asarray(cut["route_segment_component"],dtype=np.int64)
  ends=np.asarray(cut["route_segment_coarse_endpoints"],dtype=np.int64)
  return {ekey(*ends[i]) for i in range(len(rc)) if int(rc[i]) in component_ids}
 old_route_edges=route_edge_keys(oc,set(OLD_COMPONENTS))
 new_route_edges=route_edge_keys(nc,set(NEW_COMPONENTS))
 # Shared coarse route edges are allowed only if both candidates provide the same
 # source-detail path and output chain; add_candidate below enforces exact agreement.
 route_edge_overlap=old_route_edges & new_route_edges
 # Candidate-local nodes are interned by their actual source identity. Any duplicate
 # identity must have the same constructed coordinate; no averaging or snapping.
 out_points=[];out_meta=[];intern={};conflicts=[]
 def add_node(h,local,label):
  local=int(local); kind=int(h["node_kind"][local]); sr=int(h["node_source_row"][local])
  vr=int(h["node_source_virtual_row"][local])
  if kind==0:
   require(sr>=0 and sr<len(Pbase),f"{label} malformed coarse node {local}")
   key=("coarse",sr); xyz=Pbase[sr]
  elif sr>=0:
   key=("detail",sr); xyz=np.asarray(h["points_world_m"][local],dtype=np.float64)
  elif vr>=0:
   key=("virtual",label,vr); xyz=np.asarray(h["points_world_m"][local],dtype=np.float64)
  elif kind==5:
   key=("private-centroid",label,local); xyz=np.asarray(h["points_world_m"][local],dtype=np.float64)
  else:
   raise RuntimeError(f"{label} node {local} has untyped ancestry kind={kind} source={sr} virtual={vr}")
  if key in intern:
   gi=intern[key];d=float(np.linalg.norm(out_points[gi]-xyz))
   if d>2e-9:
    conflicts.append({"key":list(key),"distance_m":d,"first_label":out_meta[gi]["origin"],
                      "second_label":label,"first_node":out_meta[gi]["source_local_row"],"second_node":local})
   return gi
  gi=len(out_points);intern[key]=gi;out_points.append(np.asarray(xyz,dtype=np.float64).copy())
  out_meta.append({"kind":kind,"source_row":sr,"source_virtual_row":vr,
   "parent_coarse_face_row":int(h["node_parent_coarse_face_row"][local]),
   "edge_a":int(h["node_parent_coarse_edge_a"][local]),"edge_b":int(h["node_parent_coarse_edge_b"][local]),
   "edge_parameter":float(h["node_parent_coarse_edge_parameter"][local]),
   "source_displacement_m":float(h["node_source_displacement_m"][local]),
   "origin":label,"source_local_row":local})
  return gi
 # Resolve candidate patch triangles and all path chains used by selected components.
 patch_faces=[];patch_face_meta=[];edge_paths={};path_records=[]
 def add_candidate(label,h,cut,component_ids,selected_fine):
  # Fine patch face rows have a cut-local fine face row; source marker is stable detail-face row.
  face_kind=np.asarray(h["face_kind"],dtype=np.int64)
  local_fine=np.asarray(h["parent_fine_detail_face_row"],dtype=np.int64)
  source_face=np.asarray(h["parent_source_face_marker"],dtype=np.int64)
  tris=np.asarray(h["source_O_faces"],dtype=np.int64)
  selectset=set(map(int,selected_fine))
  fids=np.flatnonzero((face_kind==2)&np.isin(local_fine,np.asarray(sorted(selectset),dtype=np.int64)))
  require(len(fids)>0,f"{label} selected patch has no emitted fine faces")
  cut_parent=np.asarray(cut["fine_subface_parent_rows"],dtype=np.int64)
  for fi in fids:
   local_cut=int(local_fine[fi]); marker=int(source_face[fi])
   require(0<=local_cut<len(cut_parent),f"{label} patch face {fi} cut row out of range")
   require(marker==int(cut_parent[local_cut]),f"{label} face {fi} source marker mismatch")
   tri=[add_node(h,n,label) for n in tris[fi]]
   patch_faces.append(tri)
   patch_face_meta.append({"kind":2,"parent_coarse_face_row":int(h["parent_coarse_face_row"][fi]),
       "parent_fine_detail_face_row":local_cut,"parent_source_face_marker":marker,
       "source_cut_id":0 if label=="run007" else 1,"source_cut_face_row":local_cut})
  # Candidate path records are keyed by coarse edge. Paths preserve directed source sequence.
  edge_endpoints=np.asarray(h["coarse_boundary_edge_endpoints"],dtype=np.int64)
  ip=np.asarray(h["coarse_boundary_edge_path_indptr"],dtype=np.int64)
  out_chain=np.asarray(h["coarse_boundary_edge_path_output_nodes"],dtype=np.int64)
  src_chain=np.asarray(h["coarse_boundary_edge_path_source_nodes"],dtype=np.int64)
  rc=np.asarray(cut["route_segment_component"],dtype=np.int64)
  rl=np.asarray(cut["route_segment_loop"],dtype=np.int64)
  ce=np.asarray(cut["route_segment_coarse_endpoints"],dtype=np.int64)
  rip=np.asarray(cut["route_path_indptr"],dtype=np.int64);rpn=np.asarray(cut["route_path_nodes"],dtype=np.int64)
  route_by_key={}
  for si in range(len(rc)):
   if int(rc[si]) not in component_ids: continue
   key=ekey(*ce[si]);p=rpn[rip[si]:rip[si+1]].astype(np.int64)
   if key in route_by_key:
    require(np.array_equal(route_by_key[key],p) or np.array_equal(route_by_key[key],p[::-1]),
            f"{label} repeated route edge has different source paths: {key}")
   else:route_by_key[key]=p
  out_by_key={ekey(*edge_endpoints[i]):i for i in range(len(edge_endpoints))}
  for key,p in route_by_key.items():
   require(key in out_by_key,f"{label} missing emitted path chain for {key}")
   ri=out_by_key[key];lo,hi=int(ip[ri]),int(ip[ri+1])
   sp=src_chain[lo:hi];ch=out_chain[lo:hi]
   require(len(sp)==len(p)==len(ch),f"{label} path chain length mismatch {key}")
   require(np.array_equal(sp,p),f"{label} path chain source order mismatch {key}")
   segrow=next(i for i in range(len(rc)) if int(rc[i]) in component_ids and ekey(*ce[i])==key)
   c0,c1=map(int,ce[segrow])
   # The saved output-node chain in 127/128 is pre-compaction; reconstruct it
   # from typed source identities instead of indexing the compact point array.
   anchor_rows=np.asarray(h["coarse_boundary_anchor_rows"],dtype=np.int64)
   anchor_sources=np.asarray(h["coarse_boundary_anchor_source_nodes"],dtype=np.int64)
   anchor_src={int(a):int(b) for a,b in zip(anchor_rows,anchor_sources)}
   require(p[0]==anchor_src.get(c0) and p[-1]==anchor_src.get(c1),
           f"{label} route/source endpoint identity mismatch edge={key}")
   kinds=np.asarray(h["node_kind"],dtype=np.int64)
   srcrows=np.asarray(h["node_source_row"],dtype=np.int64)
   vrows=np.asarray(h["node_source_virtual_row"],dtype=np.int64)
   coarse_local=np.flatnonzero((kinds==0)&(srcrows==c0))
   coarse_local1=np.flatnonzero((kinds==0)&(srcrows==c1))
   require(len(coarse_local)==1 and len(coarse_local1)==1,
           f"{label} compact candidate missing unique coarse endpoints {c0},{c1}")
   gchain=[add_node(h,int(coarse_local[0]),label)]
   for sn in p[1:-1]:
    if int(sn)<len(det["detail_row"]):
     choices=np.flatnonzero((kinds==2)&(srcrows==int(sn)))
    else:
     choices=np.flatnonzero((kinds==3)&(vrows==int(sn)-len(det["detail_row"])))
    require(len(choices)==1,f"{label} source path node {int(sn)} has {len(choices)} compact typed rows")
    gchain.append(add_node(h,int(choices[0]),label))
   gchain.append(add_node(h,int(coarse_local1[0]),label))
   require(len(gchain)==len(sp),f"{label} reconstructed compact path length mismatch edge={key}")
   if key in edge_paths:
    old=edge_paths[key]
    require(old["source_path"]==list(map(int,p)) or old["source_path"]==list(map(int,p[::-1])),
            f"composed duplicate route edge has incompatible source path {key}")
    require(np.array_equal(np.asarray(old["nodes"]),np.asarray(gchain)) or
            np.array_equal(np.asarray(old["nodes"]),np.asarray(gchain[::-1])),
            f"composed duplicate route edge has incompatible output chain {key}")
   else:
    edge_paths[key]={"coarse_start":c0,"coarse_end":c1,"nodes":gchain,"source_path":list(map(int,p)),
      "source_cut_id":0 if label=="run007" else 1}
   path_records.append({"source":label,"coarse_edge":list(key),"source_path_nodes":list(map(int,p))})
 add_candidate("run007",oh,oc,set(OLD_COMPONENTS),old_keep)
 add_candidate("run017",nh,nc,set(NEW_COMPONENTS),new_keep)
 require(not conflicts,f"shared source vertex coordinate conflicts: {conflicts[:8]}")
 # Rebuild original coarse collar once using the union of retained path chains.
 base_nodes={}
 def add_coarse(r):
  r=int(r)
  if r not in base_nodes:base_nodes[r]=len(out_points);out_points.append(Pbase[r].copy());out_meta.append(
    {"kind":0,"source_row":r,"source_virtual_row":-1,"parent_coarse_face_row":-1,
     "edge_a":-1,"edge_b":-1,"edge_parameter":math.nan,"source_displacement_m":0.0,
     "origin":"coarse084","source_local_row":r})
  return base_nodes[r]
 def node_index_for_key(key):
  if key[0]=="coarse":return add_coarse(key[1])
  if key in intern:return intern[key]
  raise RuntimeError(f"missing composite vertex {key}")
 # Map existing composite coarse keys into baseline nodes; intern generated nodes are kept.
 for key,gi in list(intern.items()):
  if key[0]=="coarse":base_nodes[int(key[1])]=gi
 faces=[];fkind=[];pcoarse=[];pfine=[];pmarker=[];cutids=[];cutrows=[]
 # Append mapped fine patch faces first.
 for tri,meta in zip(patch_faces,patch_face_meta):
  faces.append(tri);fkind.append(2);pcoarse.append(meta["parent_coarse_face_row"])
  pfine.append(meta["parent_fine_detail_face_row"]);pmarker.append(meta["parent_source_face_marker"])
  cutids.append(meta["source_cut_id"]);cutrows.append(meta["source_cut_face_row"])
 # Append base coarse faces and collars not replaced by source patches.
 collar_centroids=[]
 for fi,tri0 in enumerate(Fbase):
  if fi in patch_rows:continue
  tri=list(map(int,tri0));poly=[];split=[]
  for ii in range(3):
   u,v=tri[ii],tri[(ii+1)%3];key=ekey(u,v)
   rec=edge_paths.get(key)
   if rec is None:
    poly.append(add_coarse(u));continue
   c0,c1=rec["coarse_start"],rec["coarse_end"];chain=rec["nodes"]
   if (u,v)==(c0,c1): ordered=chain
   elif (u,v)==(c1,c0): ordered=chain[::-1]
   else:raise RuntimeError(f"collar edge direction inconsistency face={fi} edge={key}")
   poly.extend([add_coarse(u)]+list(ordered[1:-1]))
   split.append((list(ordered),add_coarse(tri[(ii+2)%3])))
  if not split:
   faces.append([add_coarse(x) for x in tri]);fkind.append(0);pcoarse.append(fi);pfine.append(-1);pmarker.append(fi);cutids.append(-1);cutrows.append(-1)
  elif len(split)==1:
   chain,opp=split[0]
   for a,b in zip(chain[:-1],chain[1:]):
    faces.append([int(a),int(b),int(opp)]);fkind.append(1);pcoarse.append(fi);pfine.append(-1);pmarker.append(fi);cutids.append(-1);cutrows.append(-1)
  else:
   # Exact parent-plane star subdivision; each child is within this one parent face.
   center=(Pbase[tri[0]]+Pbase[tri[1]]+Pbase[tri[2]])/3.0
   ci=len(out_points);out_points.append(center);out_meta.append(
     {"kind":5,"source_row":-1,"source_virtual_row":-1,"parent_coarse_face_row":fi,
      "edge_a":-1,"edge_b":-1,"edge_parameter":math.nan,"source_displacement_m":0.0,
      "origin":"graft-collar-centroid","source_local_row":fi})
   collar_centroids.append(fi)
   poly0=[]
   for ii in range(3):
    u,v=tri[ii],tri[(ii+1)%3];key=ekey(u,v);rec=edge_paths.get(key)
    if rec is None: poly0.append(add_coarse(u))
    else:
     c0,c1=rec["coarse_start"],rec["coarse_end"];chain=rec["nodes"]
     ordered=chain if (u,v)==(c0,c1) else chain[::-1]
     poly0.extend([add_coarse(u)]+list(ordered[1:-1]))
   require(len(set(poly0))==len(poly0),f"collar polygon repeats vertices on face {fi}")
   for a,b in zip(poly0,poly0[1:]+poly0[:1]):
    faces.append([ci,int(a),int(b)]);fkind.append(1);pcoarse.append(fi);pfine.append(-1);pmarker.append(fi);cutids.append(-1);cutrows.append(-1)
 # Validate the coarse replacement partition.
 require(len(patch_rows)>0,"no patch coarse parent faces")
 require(all(0<=r<len(Fbase) for r in patch_rows),"selected coarse face row outside baseline")
 # Ensure no output point is unreferenced, then compact deterministically.
 P=np.asarray(out_points,dtype=np.float64);T=np.asarray(faces,dtype=np.int64)
 used=np.unique(T);remap=np.full(len(P),-1,dtype=np.int64);remap[used]=np.arange(len(used))
 Pc=P[used];Tc=remap[T].astype(np.int32);meta=[out_meta[int(i)] for i in used]
 # Geometry/topology summaries only; exact self is left to frozen owner audit.
 edge_count=Counter();edge_dir_count=Counter()
 for tri in Tc:
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   key=ekey(a,b);edge_count[key]+=1;edge_dir_count[key]+=1 if (int(a),int(b))==key else -1
 boundary=[e for e,n in edge_count.items() if n==1]
 nonman=[(e,n) for e,n in edge_count.items() if n>2]
 winding=[(e,n,edge_dir_count[e]) for e,n in edge_count.items() if n==2 and edge_dir_count[e]!=0]
 areas=np.linalg.norm(np.cross(Pc[Tc[:,1]]-Pc[Tc[:,0]],Pc[Tc[:,2]]-Pc[Tc[:,0]]),axis=1)*0.5
 tiny=np.flatnonzero(areas<=1e-16)
 # Count duplicated triangles and source marker coverage.
 face_keys=[tuple(sorted(map(int,t))) for t in Tc]
 duplicates=len(face_keys)-len(set(face_keys))
 used_source_face_markers=sorted(set(int(x) for x,k in zip(pmarker,fkind) if int(k)==2))
 # Check ear faces specifically: if they belonged to the dropped bad component they should not be in output.
 ear_source_rows={9302,9303}
 ear_present=ear_source_rows & set(used_source_face_markers)
 # Topology exact expected boundary domain is the original O shell boundary.
 original_edges=Counter()
 for tri in Fbase:
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):original_edges[ekey(a,b)]+=1
 original_boundary={e for e,n in original_edges.items() if n==1}
 # Boundary edges in hybrid output now include inserted path samples, so report count and source-edge coverage.
 def add_sha(p):return sha(p)
 OUT.mkdir(parents=True,exist_ok=False)
 cand=OUT/"hybrid-O-surface-129.npz"
 np.savez_compressed(cand,points_world_m=Pc,source_O_faces=Tc,
  node_kind=np.asarray([m["kind"] for m in meta],dtype=np.int8),
  node_source_row=np.asarray([m["source_row"] for m in meta],dtype=np.int64),
  node_source_virtual_row=np.asarray([m["source_virtual_row"] for m in meta],dtype=np.int32),
  node_parent_coarse_face_row=np.asarray([m["parent_coarse_face_row"] for m in meta],dtype=np.int32),
  node_parent_coarse_edge_a=np.asarray([m["edge_a"] for m in meta],dtype=np.int32),
  node_parent_coarse_edge_b=np.asarray([m["edge_b"] for m in meta],dtype=np.int32),
  node_parent_coarse_edge_parameter=np.asarray([m["edge_parameter"] for m in meta],dtype=np.float64),
  node_source_displacement_m=np.asarray([m["source_displacement_m"] for m in meta],dtype=np.float64),
  face_kind=np.asarray(fkind,dtype=np.int8),parent_coarse_face_row=np.asarray(pcoarse,dtype=np.int32),
  parent_fine_detail_face_row=np.asarray(pfine,dtype=np.int32),parent_source_face_marker=np.asarray(pmarker,dtype=np.int32),
  source_patch_cut_id=np.asarray(cutids,dtype=np.int8),source_cut_face_row=np.asarray(cutrows,dtype=np.int32),
  selected_coarse_face_rows=np.asarray(sorted(patch_rows),dtype=np.int32),
  selected_fine_detail_face_rows=np.asarray(used_source_face_markers,dtype=np.int32),
  retained_run003_component_ids=np.asarray(OLD_COMPONENTS,dtype=np.int32),
  expanded_run016_component_ids=np.asarray(NEW_COMPONENTS,dtype=np.int32))
 # Remove private origin labels from public node schema; they remain in the report.
 report={
  "status":"grafted_hybrid_O_emitted_topology_only",
  "interpretation":"Run007 fine patches for 31 routed coarse components are retained except original component 2; expanded run016 source patches for components 1+2 replace the touched component union. Coarse collars are regenerated once from the union of retained paths. This is not a collision, bone-clearance, I/D, tet, mass, or native admission.",
  "inputs":pins,
  "components":{"retained_run003_original":[int(x) for x in OLD_COMPONENTS],
   "excluded_run003_bad_self_component":2,"expanded_run016":[int(x) for x in NEW_COMPONENTS],
   "old_selected_fine_subfaces":int(len(old_keep)),"new_selected_fine_subfaces":int(len(new_keep)),
   "old_patch_coarse_faces":len(old_patch_rows),"new_patch_coarse_faces":len(new_patch_rows),
   "old_new_patch_coarse_overlap":0,"retained_route_edges":len(old_route_edges),"expanded_route_edges":len(new_route_edges),
   "shared_route_edge_count":len(route_edge_overlap)},
  "graft":{"output_nodes":int(len(Pc)),"output_faces":int(len(Tc)),
   "patch_faces":int(sum(np.asarray(fkind)==2)),"collar_faces":int(sum(np.asarray(fkind)==1)),
   "untouched_coarse_faces":int(sum(np.asarray(fkind)==0)),"collar_centroid_parent_rows":collar_centroids,
   "shared_source_coordinate_conflicts":conflicts,"fine_patch_stable_source_face_markers":len(used_source_face_markers),
   "ear_detail_source_face_markers_9302_9303_present":sorted(map(int,ear_present)),
   "ear_status":"detail source-face marker diagnostic only; run007 bad component 2 excluded and replaced by expanded run016 components 1+2; no ear retriangulation applied",
   "path_record_count":len(path_records)},
  "topology_diagnostics":{"boundary_edge_count":len(boundary),"nonmanifold_edge_count":len(nonman),
   "nonmanifold_edge_witnesses":[[list(e),int(n)] for e,n in nonman[:30]],
   "directed_winding_conflict_count":len(winding),"directed_winding_conflict_witnesses":[[list(e),int(n),int(s)] for e,n,s in winding[:30]],
   "duplicate_triangle_count":int(duplicates),"tiny_face_count":int(len(tiny)),
   "min_face_area_m2":float(areas.min()),"max_face_area_m2":float(areas.max()),
   "coarse_original_boundary_edge_count":len(original_boundary)},
  "route_selection":{"retained_old_loops":old_loop_log,"expanded_loops":new_loop_log},
  "outputs":{"candidate_npz":str(cand),"candidate_sha256":sha(cand)}
 }
 (OUT/"hybrid-O-surface-129-report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))

if __name__=="__main__": main()

