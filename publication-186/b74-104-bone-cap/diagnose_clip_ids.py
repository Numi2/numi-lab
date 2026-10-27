import json,hashlib,ast,importlib.util
from pathlib import Path
from collections import Counter,defaultdict,deque
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638'
O=S/'per-bank-workers-20261027/b74-104/steiner-multinode-005-bone-cap'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20222/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
import sys
h=module('bankhelper',Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py'))
C=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz';Z=np.load(C,allow_pickle=False);X=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
W=json.load(open(S/'per-bank-workers-20261027/b74-104/baseline-witnesses.json'));G=json.load(open(S/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided/segment-graph-report.json'));RPT=json.load(open(O/'cap-planar-report.json'))
old=list(map(int,W['skin_child_faces']));node_bary={}
for fr in old:
 for n,b in zip(F[fr],Z['refinement_face_parent_barycentric'][fr]):node_bary[int(n)]=np.asarray(b,float)
corners=[next(n for n,b in node_bary.items() if np.array_equal(b,np.eye(3)[i])) for i in range(3)]
EVID=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight')
T,ids,gids,bodies,M,off,pos,keys,pins,_=h.read_bank_inputs(EVID,E)
B=np.asarray(T[(ids[:,0]==74)&(ids[:,1]==104)],np.float32)
R=M[74];Ooff=off[74];P=pos[74];base_local=(X.astype(np.float64)-P)@R.T-Ooff;corner=base_local[np.asarray(corners)]
n=np.cross(corner[1]-corner[0],corner[2]-corner[0]);n/=np.linalg.norm(n)
# Canonical mesh vertices are exact Float32 bit-pattern coordinates.
vmap={};V=[];faces=[]
for tri in B:
 row=[]
 for q in tri:
  key=tuple(np.asarray(q,np.float32).view(np.uint32).tolist())
  if key not in vmap:vmap[key]=len(V);V.append(np.asarray(q,np.float64))
  row.append(vmap[key])
 faces.append(row)
V=np.asarray(V);faces=np.asarray(faces,np.int64);dv=(V-corner[0])@n
ec=defaultdict(list);de=defaultdict(list);ef=defaultdict(list);facekeys=Counter();areas=[]
for fi,f in enumerate(faces):
 facekeys[tuple(sorted(map(int,f)))]+=1
 areas.append(float(np.linalg.norm(np.cross(B[fi,1]-B[fi,0],B[fi,2]-B[fi,0]))*0.5))
 for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
  key=tuple(sorted((int(a),int(b))));ec[key].append(fi);de[key].append((int(a),int(b)))
  ef[key].append(fi)
# Component count of actual bone input.
adj=[set() for _ in faces]
for e,fs in ef.items():
 if len(fs)==2:adj[fs[0]].add(fs[1]);adj[fs[1]].add(fs[0])
unseen=set(range(len(faces)));bone_comps=[]
while unseen:
 a=min(unseen);q=[a];unseen.remove(a);co=[]
 while q:
  u=q.pop();co.append(u)
  for v in adj[u]:
   if v in unseen:unseen.remove(v);q.append(v)
 bone_comps.append(co)
gp=np.asarray([np.asarray(b)@base_local[np.asarray(corners)] for b in G['intersection_points_barycentric']],float)
def segdist(p,a,b):
 ab=b-a;den=float(ab@ab);t=0 if den==0 else float((p-a)@ab/den);t=min(1,max(0,t));return float(np.linalg.norm(p-(a+t*ab)))
def token_coord(tok):
 if tok[0]=='c':return gp[int(tok[1])]
 if tok[0]=='v':return V[int(tok[1])]
 if tok[0]=='x':
  a,b=map(int,tok[1:]);da=float(dv[a]);db=float(dv[b])
  if abs(da)<=2e-9:return V[a]
  if abs(db)<=2e-9:return V[b]
  return V[a]+(da/(da-db))*(V[b]-V[a])
 raise ValueError(tok)
extra_summary=[]
for side in RPT['boundary_side_results']:
 main=max(side['components'],key=lambda c:c['face_count'])
 edges=[ast.literal_eval(x) for x in main['boundary_mismatch']['extra']]
 sample=[]
 allx=[]
 for e in edges:
  vals=[]
  for tok in e:
   q=token_coord(tok);allx.append(q)
   dp=[float(np.linalg.norm(q-p)) for p in gp]
   ds=[segdist(q,gp[a],gp[b]) for a,b in G['segments_point_indices']]
   vals.append({'token':list(tok),'nearest_contour_vertex_m':min(dp),'nearest_contour_segment_m':min(ds),'source_plane_signed_distance_m':float((q-corner[0])@n)})
  sample.append(vals)
 extra_summary.append({'side_sign':side['side_sign'],'largest_component_faces':main['face_count'],'boundary_edge_count':main['boundary_edge_count'],'missing_target_edges':len(main['boundary_mismatch']['missing']),'extra_edges':len(edges),'extra_edges_sample':sample[:6],'extra_token_point_count':len(allx),'max_extra_point_nearest_contour_segment_m':max((min(segdist(q,gp[a],gp[b]) for a,b in G['segments_point_indices']) for q in allx),default=None),'min_extra_point_nearest_contour_vertex_m':min((min(np.linalg.norm(q-p) for p in gp) for q in allx),default=None)})
source=O/'clip_bone_cap.py';source_lines=source.read_text().splitlines()
insertion_lines=[i+1 for i,l in enumerate(source_lines) if 'interior=[(' in l or 'clean=clean[:qi+1]+interior+clean[qj:]' in l]
diag={'schema':'numi.human.b74_104_clip_id_diagnostic.v1','source_report_path':str(O/'cap-planar-report.json'),'source_report_sha256':hashlib.sha256((O/'cap-planar-report.json').read_bytes()).hexdigest(),'clip_script_path':str(source),'clip_script_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'exact_segment_graph_sha256':hashlib.sha256((S/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided/segment-graph-report.json').read_bytes()).hexdigest(),'bone_triangle_count':len(B),'canonical_float32_vertex_count':len(V),'canonical_undirected_edge_count':len(ec),'edge_incidence_histogram':{str(k):v for k,v in Counter(map(len,ec.values())).items()},'all_bone_edges_incidence_two':all(len(q)==2 for q in ec.values()),'all_directed_edges_opposed':all(len(q)==2 and q[0]==(q[1][1],q[1][0]) for q in de.values()),'bone_face_connected_components':len(bone_comps),'bone_face_duplicate_count':sum(v-1 for v in facekeys.values() if v>1),'zero_area_bone_face_count':sum(a<=1e-18 for a in areas),'minimum_bone_face_area_m2':min(areas),'maximum_exact_contour_plane_residual_m':G['max_projection_residual_m'],'plane_vertex_tolerance_m':RPT['plane_vertex_tolerance_m'],'endpoint_matching_tolerance_m':1e-6,'clip_script_endpoint_mapping_lines':insertion_lines,'largest_component_extra_endpoint_distances':extra_summary,'diagnosis':'The clip source inserts only intermediate contour c-tokens; matched cut-edge endpoints remain their x/v tokens. Thus the target contour edge set (c-c) cannot match the clipped boundary even where geometry coincides. Canonical Float32 input vertices and bone-edge IDs are shared and incidence-two; the exact contour lies within 8.93e-9 m of the single source plane, so a real plane/skin discrepancy is far below the unchanged 1e-6 m endpoint match tolerance. The missing contour is an endpoint identity/welding bug, not duplicate or degenerate input bone faces.'}
(O/'clip-diagnostic.json').write_text(json.dumps(diag,indent=2)+'\n')
print(json.dumps({k:v for k,v in diag.items() if k!='largest_component_extra_endpoint_distances'},indent=2))
print(json.dumps({'component_endpoint_diagnostics':extra_summary},indent=2))

