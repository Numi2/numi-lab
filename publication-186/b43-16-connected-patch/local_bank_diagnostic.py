#!/usr/bin/env python3
"""Local topology-first diagnostic for bank 43/16 on the 653 checkpoint."""
from pathlib import Path
from collections import defaultdict,deque,Counter
import hashlib,importlib.util,json,sys,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
OUT=S/'per-bank-workers-20261027/b43-16/attempt-001/local-diagnostic.json'
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def uniq(xs):return sorted(set(map(int,xs)))
def point_seg_dist(p,a,b):
 v=b-a;d=float(v@v);t=0.0 if d==0 else min(1.0,max(0.0,float((p-a)@v/d)));return float(np.linalg.norm(p-(a+t*v)))
def point_tri_dist(p,t):
 a,b,c=t;ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=float(n@n)
 if nn<1e-24:return min(point_seg_dist(p,a,b),point_seg_dist(p,b,c),point_seg_dist(p,c,a))
 q=p-(n@(p-a))/nn*n;v0=ab;v1=ac;v2=q-a;d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;u=1-v-w
 if min(u,v,w)>=-1e-12:return float(np.linalg.norm(p-q))
 return min(point_seg_dist(p,a,b),point_seg_dist(p,b,c),point_seg_dist(p,c,a))
def winding(p,tri):
 a=tri[:,0]-p;b=tri[:,1]-p;c=tri[:,2]-p;la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
 num=np.einsum('ij,ij->i',a,np.cross(b,c));den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
 return float(np.sum(2*np.arctan2(num,den))/(4*np.pi))
started=time.monotonic();audit=mod('bank43_local_audit',AUDIT);helper=mod('bank43_local_helper',HELPER)
with np.load(C,allow_pickle=False) as z: Z={k:z[k].copy() for k in z.files}
P=Z['points_world_m'].astype(np.float64);F=Z['source_O_faces'].astype(np.int64)
base=json.loads(A.read_text());assert base['complete'] and base['candidate_sha256']==sha(C) and base['bank_count']==185
bank=next(b for b in base['banks'] if (b['body'],b['stable_bone_id'])==(43,16));pairs=[tuple(map(int,p)) for p in bank['result']['triangle_pairs']];cross=sorted({sf for sf,bf in pairs});bone_cross=sorted({bf for sf,bf in pairs})
# Full pinned bank input is loaded for geometry/topology only; no audit is run over other banks.
bankdir=audit.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,bankkeys,pins,unused=helper.read_bank_inputs(bankdir,audit.E)
mask=(ids[:,0]==43)&(ids[:,1]==16);BT=T[mask].astype(np.float64);BID=ids[mask,3].astype(np.int64);bidx={int(fid):i for i,fid in enumerate(BID)}
# Recover exact bone topological vertex IDs by bitwise matching the pinned Float32 triangle vertices to the pinned node table.
nodes_path=bankdir/'exact-emitted.nodes.csv';nodes=np.loadtxt(nodes_path,delimiter=',',skiprows=1);node_gids=nodes[:,1].astype(np.int64);node_f32=nodes[:,5:8].astype(np.float32);node_keys=[tuple(map(int,row.view(np.uint32))) for row in node_f32];assert len(set(node_keys))==len(node_keys)
node_key_to_gid={k:int(g) for k,g in zip(node_keys,node_gids)}
BVID=[];unmapped=[]
for j,t in enumerate(BT):
 row=[]
 for v in t:
  k=tuple(map(int,np.asarray(v,dtype=np.float32).view(np.uint32)))
  if k not in node_key_to_gid:unmapped.append([int(BID[j]),k]);row.append(-1)
  else:row.append(node_key_to_gid[k])
 BVID.append(row)
BVID=np.asarray(BVID,dtype=np.int64)
bone_edge_faces=defaultdict(list);bone_node_faces=defaultdict(list)
for i,t in enumerate(BVID):
 for v in t:bone_node_faces[int(v)].append(i)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):bone_edge_faces[tuple(sorted((int(u),int(v))))].append(i)
boundary_bone_edges=[(e,len(v)) for e,v in bone_edge_faces.items() if len(v)==1];nonmanifold_bone_edges=[(e,len(v)) for e,v in bone_edge_faces.items() if len(v)>2]
# Crossing skin face components by shared mesh edges; also identify vertex-connected components.
edgefaces=defaultdict(list);nodefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t:nodefaces[int(v)].append(fi)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
def components(rows,by_vertex=False):
 unseen=set(rows);comps=[]
 while unseen:
  st=min(unseen);unseen.remove(st);q=deque([st]);c=[]
  while q:
   f=q.popleft();c.append(f);cand=set()
   if by_vertex:
    for v in F[f]:cand.update(nodefaces[int(v)])
   else:
    for u,v in ((F[f,0],F[f,1]),(F[f,1],F[f,2]),(F[f,2],F[f,0])):cand.update(edgefaces[tuple(sorted((int(u),int(v))))])
   for g in cand&unseen:unseen.remove(g);q.append(g)
  comps.append(sorted(c))
 return comps
cc=components(cross);vc=components(cross,True)
# Merge components whose one-rings overlap so every fixed patch has an unambiguous boundary.
patches=[]
for comp in cc:
 patch=set(comp)
 for f in comp:
  for v in F[f]:patch.update(nodefaces[int(v)])
 patches.append(patch)
changed=True
while changed:
 changed=False
 out=[]
 while patches:
  a0=patches.pop(0);merged=True
  while merged:
   merged=False
   for i,b0 in enumerate(patches):
    if a0 & b0:
     a0|=b0;patches.pop(i);merged=True;changed=True;break
  out.append(a0)
 patches=out
# Ancestry arrays.
ancptr=Z['face_source_ancestry_indptr'];anc=Z['face_source_ancestry_face_rows'];detptr=Z['face_fine_detail_ancestry_indptr'];det=Z['face_fine_detail_ancestry_rows'];subptr=Z['face_fine_subface_ancestry_indptr'];sub=Z['face_fine_subface_ancestry_rows'];parent=Z['refinement_face_parent_face_row']
local_tri=((P-pos[43])@M[43].T-off[43])
patch_reports=[];all_patch_nodes=set()
for pi,patch in enumerate(patches):
 fs=sorted(patch);cross_fs=sorted(set(fs)&set(cross));all_patch_nodes.update(map(int,F[fs].reshape(-1)))
 p_edges=defaultdict(list)
 for f in fs:
  t=F[f]
  for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):p_edges[tuple(sorted((int(u),int(v))))].append(f)
 boundary=[]
 for e,inside in p_edges.items():
  outs=[g for g in edgefaces[e] if g not in patch]
  if outs or len(edgefaces[e])==1:boundary.append((e,outs))
 bg=defaultdict(list)
 for e,_ in boundary:bg[e[0]].append(e[1]);bg[e[1]].append(e[0])
 bcomps=[];unseen=set(bg)
 while unseen:
  st=min(unseen);unseen.remove(st);q=[st];cc0=[]
  while q:
   v=q.pop();cc0.append(v)
   for w in bg[v]:
    if w in unseen:unseen.remove(w);q.append(w)
  bcomps.append(cc0)
 # Crossing face ancestry map and boundary-node local bone relation.
 def ancestry(faces,ptr,vals):return uniq(v for f in faces for v in vals[ptr[f]:ptr[f+1]])
 for x in fs:pass
 bnodes=sorted(bg)
 boundary_geom=[]
 for n in bnodes:
  dist=min(point_tri_dist(local_tri[n],BT) for _ in [0]) if False else min(point_tri_dist(local_tri[n],tr) for tr in BT)
  w=winding(local_tri[n],BT)
  boundary_geom.append({'node_row':n,'distance_to_bone_local_mm':float(dist*1000),'winding':w,'inside_by_abs_winding_gt_half':bool(abs(w)>0.5)})
 crossing_nodes=sorted({int(v) for f in cross_fs for v in F[f]});
 inside_cross=[]
 for n in crossing_nodes:
  w=winding(local_tri[n],BT)
  if abs(w)>0.5:
   d=min(point_tri_dist(local_tri[n],tr) for tr in BT)
   inside_cross.append({'node_row':n,'winding':w,'nearest_bone_distance_mm':float(d*1000)})
 patch_reports.append({'patch_component':pi,'crossing_skin_faces':cross_fs,'crossing_pair_count':sum(1 for sf,bf in pairs if sf in set(cross_fs)),'crossing_bone_faces':uniq(bf for sf,bf in pairs if sf in set(cross_fs)),'patch_face_count':len(fs),'patch_node_count':len(set(map(int,F[fs].reshape(-1)))),'crossing_node_count':len(crossing_nodes),'crossing_source_ancestry_rows':ancestry(cross_fs,ancptr,anc),'crossing_fine_detail_ancestry_rows':ancestry(cross_fs,detptr,det),'crossing_fine_subface_ancestry_rows':ancestry(cross_fs,subptr,sub),'crossing_immediate_parent_face_rows':uniq(parent[cross_fs]),'patch_outer_boundary_edge_count':len(boundary),'patch_outer_boundary_node_count':len(bnodes),'boundary_loop_count':len(bcomps),'boundary_degree_histogram':dict(Counter(len(bg[n]) for n in bg)),'noncrossing_faces_adjacent_to_patch_boundary':len({g for e,outs in boundary for g in outs}),'boundary_nodes_inside_bone_count':sum(x['inside_by_abs_winding_gt_half'] for x in boundary_geom),'boundary_node_min_clearance_mm':min((x['distance_to_bone_local_mm'] for x in boundary_geom),default=None),'boundary_node_max_clearance_mm':max((x['distance_to_bone_local_mm'] for x in boundary_geom),default=None),'boundary_nodes':boundary_geom,'inside_crossing_nodes':inside_cross})
# Topology of the matched bone faces: component IDs by shared stable-node edges.
active_bone_local=[bidx[x] for x in bone_cross];bone_comps=[];unseen=set(active_bone_local)
while unseen:
 st=min(unseen);unseen.remove(st);q=deque([st]);comp=[]
 while q:
  i=q.popleft();comp.append(int(BID[i]));
  for a0,b0 in ((BVID[i,0],BVID[i,1]),(BVID[i,1],BVID[i,2]),(BVID[i,2],BVID[i,0])):
   for j in bone_edge_faces[tuple(sorted((int(a0),int(b0))))]:
    if j in unseen:unseen.remove(j);q.append(j)
 bone_comps.append(sorted(comp))
report={'schema':'numi.human.bank43_16.topology_first_diagnostic.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_path':str(A),'audit159_sha256':sha(A),'audit_complete':bool(base['complete']),'typed_bank_count':int(base['bank_count']),'typed_bank':[43,16],'exact_cross_pair_count':len(pairs),'unique_crossing_skin_face_count':len(cross),'crossing_skin_face_rows':cross,'unique_crossing_bone_face_count':len(bone_cross),'crossing_bone_face_rows':bone_cross,'crossing_skin_edge_connected_components':len(cc),'crossing_skin_vertex_connected_components':len(vc),'crossing_skin_components_edge':[c for c in cc],'one_ring_merged_patch_count':len(patches),'patches':patch_reports,'bone_mesh_topology':{'face_count':int(len(BT)),'topological_vertex_ids_recovered_by_bitwise_pin_match':int(len(BVID.reshape(-1))-np.count_nonzero(BVID<0)),'unmapped_vertex_corners':len(unmapped),'boundary_edge_count':len(boundary_bone_edges),'nonmanifold_edge_count':len(nonmanifold_bone_edges),'crossing_bone_face_components':bone_comps},'inputs':{'bank_triangle_export_pins':pins,'bank_local_frame_rule':'skin local=(world-position)@M.T-offset; exact emitted bank triangle float32 values promoted to float64','coordinate_bin_endpoint_welding_used':False},'limitation':['Topology and local point containment/clearance diagnostics only. Existing audit-159 supplies exact crossing pairs; no new full or all-bank audit, self audit, patch construction, or physical/native admission.'],'elapsed_seconds':time.monotonic()-started}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('patches','crossing_skin_components_edge')},indent=2,allow_nan=False))
print('patch summaries',json.dumps([{k:v for k,v in p.items() if k not in ('boundary_nodes','inside_crossing_nodes')} for p in patch_reports],indent=2,allow_nan=False))
