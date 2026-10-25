#!/Users/n/numi-human-prep-venv-20261005/bin/python
"""Trace baseline-103 hit edge (O329,O340) through the frozen NHSKIN surface graph.

This is a read-only, local graph/path analysis. It uses exact existing edge-hit
witnesses and the pinned 075-to-084 transfer; it does not launch a bone query.
"""
from __future__ import annotations
import csv, hashlib, json, struct
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra

WORK=Path('/Users/n/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BASE=WORK/'connected-cell-witness-103/run-001/baseline'
COMPACT=WORK/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
TRANSFER_DIR=WORK/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002'
DETAIL=E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'
ROWS=E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv'
COARSE_FACES=E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-coarse-faces.csv'
MARKER_MAP=WORK/'whole-reference-tetgen-084-bounded/facet-marker-map.csv'
REPORT1=WORK/'digit-source-diagnosis-110/report.json'
SCRIPT1=WORK/'digit-source-diagnosis-110/analyze_digit_edges_110.py'
OUT=WORK/'digit-source-diagnosis-110/report-002.json'
TARGET_EDGE=(329,340)
EXPECTED={
 'baseline_edge_report':'41c8b9625e99ae6f5a4b4f3d909e847c906c53adc7523b221e0f2373faa7c2ee',
 'compact':'5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
 'transfer_matrix':'55329313bf9c2c3b613fcd7eb52dd9b7b453520dd1b4a14e0e0d6ac457ce8519',
 'detail_payload':'7456bd4e1c646207ac69978a31a0298464d1e157b96b6ce528e688d736fee9314',
 'transfer_rows':'71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14',
}
# Correct the detail payload SHA from the immutable source receipt; fail closed below.
EXPECTED['detail_payload']='7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def pin(name,p):
 h=sha(p)
 if h!=EXPECTED[name]:raise RuntimeError(f'{name} SHA mismatch {h} != {EXPECTED[name]}')
 return h
def parse_detail(path):
 raw=path.read_bytes()
 magic,abi,nb,nv,ni,_=struct.unpack_from('<8s5I',raw)
 if magic!=b'NHSKIN1\0' or abi!=5 or ni%3:raise RuntimeError('unexpected detail NHSKIN ABI')
 vo=60+nb*36;io=vo+nv*56;wo=io+ni*4
 if len(raw)!=wo+nv*nb*4:raise RuntimeError('NHSKIN header length mismatch')
 f=np.frombuffer(raw,dtype='<u4',count=ni,offset=io).reshape(-1,3).astype(np.int64,copy=True)
 return f,{'abi':int(abi),'binding_count':int(nb),'vertex_count':int(nv),'face_count':int(ni//3)}
def dist_point_polyline(p,poly):
 best=float('inf')
 for a,b in zip(poly[:-1],poly[1:]):
  v=b-a;d=float(v@v)
  t=0.0 if d==0 else float(np.clip((p-a)@v/d,0.0,1.0))
  best=min(best,float(np.linalg.norm(p-(a+t*v))))
 return best

edgepath=BASE/'tet-edge-report.json'
edgehash=pin('baseline_edge_report',edgepath)
compacthash=pin('compact',COMPACT)
transferhash=pin('transfer_matrix',TRANSFER_DIR/'Wnew-detail-to-084-nodes.npz')
detailhash=pin('detail_payload',DETAIL)
rowhash=pin('transfer_rows',ROWS)
with edgepath.open() as f:edge_report=json.load(f)
with np.load(COMPACT,allow_pickle=False) as z:
 ref=np.asarray(z['reference_points_m'],dtype=np.float64)
 current=np.asarray(z['current037_points_m_f32'])
 srcrows=np.asarray(z['source_input_row'],dtype=np.int64)
 globalids=np.asarray(z['source_global_node_ids'],dtype=np.int64)
 source_o_faces=np.asarray(z['source_O_faces'],dtype=np.int64)
 surf_faces=np.asarray(z['surface_faces_zero_based'],dtype=np.int64)
 surf_markers=np.asarray(z['surface_face_markers'],dtype=np.int64)
points=np.load(BASE/'candidate.npz',allow_pickle=False)['points_world_f32']
if not np.array_equal(points,current):raise RuntimeError('103 query candidate no longer bitwise matches compact 084 current037 positions')
if not np.array_equal(srcrows[:7233],np.arange(7233)):raise RuntimeError('source row order changed')
Wdata=np.load(TRANSFER_DIR/'Wnew-detail-to-084-nodes.npz',allow_pickle=False)
W=csr_matrix((Wdata['data'],Wdata['indices'],Wdata['indptr']),shape=tuple(int(x) for x in Wdata['shape']))
detail_faces,detail_header=parse_detail(DETAIL)

# Confirm the two actual output O shell facets and their source face ancestry.
marker_rows={}
with MARKER_MAP.open(newline='') as f:
 for r in csv.DictReader(f):marker_rows[int(r['marker'])]=r
output_faces=[]
for outrow,(tri,marker) in enumerate(zip(surf_faces,surf_markers)):
 m=marker_rows[int(marker)]
 if m['surface']=='O' and set(TARGET_EDGE)<=set(map(int,tri)):
  output_faces.append({'output_surface_face_row':outrow,'triangle_output_rows':list(map(int,tri)),
    'marker':int(marker),'source_face_row':int(m['source_face_row']),
    'marker_source_vertex_ids':list(map(int,m['vertex_ids_zero_based'].split()))})
if len(output_faces)!=2:raise RuntimeError(f'expected two O surface faces on edge, found {len(output_faces)}')
parent_rows=sorted({x['source_face_row'] for x in output_faces})
coarse_face_rows=[]
with COARSE_FACES.open(newline='') as f:
 for r in csv.DictReader(f):coarse_face_rows.append([int(r[f'vertex{i}']) for i in range(3)])
for fr in parent_rows:
 if not set(TARGET_EDGE)<=set(coarse_face_rows[fr]):raise RuntimeError(f'parent face {fr} does not contain target edge')
if not all(np.array_equal(np.sort(source_o_faces[fr]),np.sort(coarse_face_rows[fr])) for fr in parent_rows):
 raise RuntimeError('compact source_O_faces and accepted coarse face parentage disagree')

# Read the accepted transfer rows once, retaining donor/source-face lineage.
transfer_rows=[]
with ROWS.open(newline='') as f:transfer_rows=list(csv.DictReader(f))
if len(transfer_rows)!=W.shape[0]:raise RuntimeError('transfer rows and W detail row count differ')
detail_world=np.asarray([[float(r[f'detail_rest_world_{a}_m']) for a in 'xyz'] for r in transfer_rows],dtype=np.float64)
# Check that transfer parent supports and active donor metadata agree with W.
patch_vertices=set()
per_parent=Counter()
for r in transfer_rows:
 row=int(r['detail_row'])
 if int(r['active']) and int(r['selected_face_row']) in parent_rows:
  lo,hi=W.indptr[row:row+2]
  supp=set(map(int,W.indices[lo:hi][np.abs(W.data[lo:hi])>1e-10]))
  fr=int(r['selected_face_row'])
  if not supp or not supp<=set(coarse_face_rows[fr]):
   raise RuntimeError(f'detail row {row} support does not lie in selected source face {fr}: {sorted(supp)}')
  patch_vertices.add(row);per_parent[fr]+=1
# O anchor rows must be exact one-hot rows under W.
def direct_detail_anchor(coarse_row):
 found=[]
 for i in range(W.shape[0]):
  lo,hi=W.indptr[i:i+2]
  if hi-lo==1 and int(W.indices[lo])==coarse_row and abs(float(W.data[lo])-1.0)<1e-7:
   found.append(i)
 if len(found)!=1:raise RuntimeError(f'expected one direct detail anchor for coarse O row {coarse_row}, got {found}')
 return found[0]
start=direct_detail_anchor(TARGET_EDGE[0]);goal=direct_detail_anchor(TARGET_EDGE[1])
if start not in patch_vertices or goal not in patch_vertices:raise RuntimeError('direct anchors absent from two-face detail patch')

# Build the detailed face-edge graph restricted to W supports inside either
# exact parent triangle. This is a local graph operation, not a collision query.
edge_set=set()
for tri in detail_faces:
 for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
  a=int(a);b=int(b)
  if a not in patch_vertices or b not in patch_vertices:continue
  union=set()
  for r in (a,b):
   lo,hi=W.indptr[r:r+2]
   union.update(map(int,W.indices[lo:hi][np.abs(W.data[lo:hi])>1e-10]))
  if any(union<=set(coarse_face_rows[fr]) for fr in parent_rows):edge_set.add(tuple(sorted((a,b))))
if not edge_set:raise RuntimeError('no detailed surface edges in parent patch')
patch_rows=sorted(patch_vertices);local={r:i for i,r in enumerate(patch_rows)}
rr=[];cc=[];dd=[]
for a,b in edge_set:
 length=float(np.linalg.norm(detail_world[a]-detail_world[b]))
 if length<=0:raise RuntimeError('zero-length detailed surface edge')
 rr.extend((local[a],local[b]));cc.extend((local[b],local[a]));dd.extend((length,length))
G=coo_matrix((dd,(rr,cc)),shape=(len(patch_rows),len(patch_rows))).tocsr()
source_local,target_local=local[start],local[goal]
dist,prev=dijkstra(G,directed=False,indices=source_local,return_predecessors=True)
if not np.isfinite(dist[target_local]):raise RuntimeError('direct O anchors are disconnected in detailed two-face patch')
path=[];cur=target_local
while cur!=source_local:
 if cur<0:raise RuntimeError('broken predecessor chain')
 path.append(patch_rows[cur]);cur=int(prev[cur])
path.append(start);path=path[::-1]
path_xyz=detail_world[np.asarray(path,dtype=np.int64)]
chord_a=detail_world[start];chord_b=detail_world[goal]
chord=chord_b-chord_a;chord_len=float(np.linalg.norm(chord));unit=chord/chord_len
pt=path_xyz-chord_a
projected_t=(pt@unit)/chord_len
lateral=pt-(pt@unit)[:,None]*unit
lateral_mm=np.linalg.norm(lateral,axis=1)*1000
# Signed side of the chord in source parent face 670, to distinguish the two
# adjacent surface sectors. Use the actual parent face normal from reference R.
face_a=coarse_face_rows[parent_rows[0]]
pa,pb,pc=ref[np.asarray(face_a,dtype=np.int64)]
n=np.cross(pb-pa,pc-pa);n/=np.linalg.norm(n)
side=np.cross(n,unit);side/=np.linalg.norm(side)
signed_side_mm=((path_xyz-chord_a)@side)*1000
path_face_counts=Counter(int(transfer_rows[r]['selected_face_row']) for r in path)
path_detail_donors=Counter(int(transfer_rows[r]['detail_donor_body']) for r in path)
path_data=[]
for k,r in enumerate(path):
 path_data.append({'detail_row':int(r),'detail_donor_body':int(transfer_rows[r]['detail_donor_body']),
   'transfer_donor_body':int(transfer_rows[r]['transfer_donor_body']),
   'selected_source_face_row':int(transfer_rows[r]['selected_face_row']),
   'world_rest_m':[float(x) for x in detail_world[r]],
   'chord_parameter':float(projected_t[k]),'signed_side_from_parent_face0_mm':float(signed_side_mm[k]),
   'distance_to_chord_mm':float(lateral_mm[k]),
   'next_edge_length_mm':(float(np.linalg.norm(path_xyz[k+1]-path_xyz[k])*1000) if k+1<len(path) else None)})

# Verify transfer matrix represents source rows on the same physical coarse frame.
Wref=np.asarray(W@ref)
active=np.diff(W.indptr)>0
map_residual=np.linalg.norm(Wref[active]-detail_world[active],axis=1)
if np.max(map_residual)>2e-5:raise RuntimeError(f'W/reference transfer differs from detail source world by {np.max(map_residual)} m')
# Exact saved hit witnesses for this coarse edge and target bank.
edge_witnesses=[]
for w in edge_report['witnesses']:
 if tuple(sorted(map(int,w['endpoint_query_rows'])))==TARGET_EDGE and int(w['body'])==66 and int(w['bone_stable_id'])==100:
  t=float(w['exact_result']['t']);hit=points[TARGET_EDGE[0]].astype(np.float64)+t*(points[TARGET_EDGE[1]].astype(np.float64)-points[TARGET_EDGE[0]].astype(np.float64))
  edge_witnesses.append({'body':int(w['body']),'bone_stable_id':int(w['bone_stable_id']),
   'source_record_index':int(w['source_record_index']),'triangle_stable_id':int(w['triangle_stable_id']),
   'exact_t':t,'barycentric':list(map(float,w['exact_result']['barycentric'])),'world_hit_m':hit.tolist()})

# All baseline hit groups for the four target anatomical bank identities.
body_map={65:{'name':'midph3_r','stable_id':99,'bone_source_label':'FJ3306'},
          66:{'name':'distph3_r','stable_id':100,'bone_source_label':'FJ3196'},
          110:{'name':'midph2_l','stable_id':115,'bone_source_label':'FJ3296'},
          111:{'name':'distph2_l','stable_id':116,'bone_source_label':'FJ3183'}}
bank_layer=defaultdict(lambda:{'groups':0,'unique_edges':set(),'triangle_witnesses':0})
bank_oo=[]
for w in edge_report['witnesses']:
 u,v=map(int,w['endpoint_query_rows']);edge=tuple(sorted((u,v)))
 if (int(w['body']),int(w['bone_stable_id'])) not in {(b,m['stable_id']) for b,m in body_map.items()}:continue
 # report v1 has all exact group classes; use compact source-row labels
 def lay(r):
  s=int(srcrows[r]);return 'generated' if s<0 else ('O' if s<2411 else 'I' if s<4822 else 'D')
 lu,lv=lay(u),lay(v);cat='generated' if 'generated' in (lu,lv) else '-'.join(sorted((lu,lv),key={'O':0,'I':1,'D':2}.get))
 k=(int(w['body']),int(w['bone_stable_id']),edge,cat)
 bank_layer[k]['groups']=1;bank_layer[k]['unique_edges'].add(edge);bank_layer[k]['triangle_witnesses']+=1
 if cat=='O-O':bank_oo.append((edge,int(w['body']),int(w['bone_stable_id']),int(w['triangle_stable_id'])))
# collapse witness multiplicity to exact edge-bank groups
uniq_bank_groups={ (k[0],k[1],k[2],k[3]) for k in bank_layer }
target_bank_summary={}
for body,meta in body_map.items():
 rows_for=[k for k in uniq_bank_groups if k[0]==body and k[1]==meta['stable_id']]
 cats=Counter(k[3] for k in rows_for)
 target_bank_summary[str(body)]={'anatomical_label':meta['name'],'source_bone_label':meta['bone_source_label'],
   'stable_id':meta['stable_id'],'unique_edge_bank_groups':len(rows_for),'groups_by_layer_pair':dict(sorted(cats.items()))}

# Geometry measures for the coarse chord and its fine path.
coarse_current_chord=float(np.linalg.norm(current[TARGET_EDGE[0]].astype(np.float64)-current[TARGET_EDGE[1]].astype(np.float64)))
source_face_rows_for_edge=[int(i) for i,t in enumerate(source_o_faces) if set(TARGET_EDGE)<=set(map(int,t))]
if sorted(source_face_rows_for_edge)!=parent_rows:raise RuntimeError('source O face rows disagree with output marker parents')
# Edge/bank exact witness hit points are measured relative to the detailed path,
# but this is not an all-bank clearance result.
hit_to_path=[]
for wh in edge_witnesses:
 hp=np.asarray(wh['world_hit_m'],dtype=np.float64)
 hit_to_path.append({'triangle_stable_id':wh['triangle_stable_id'],'distance_to_detailed_path_mm':dist_point_polyline(hp,path_xyz)*1000})

report={
 'status':'READ_ONLY_LOCAL_SOURCE_PATH_DIAGNOSIS',
 'scope':'Exact baseline103 O-O edge/bank witness plus two-face detailed NHSKIN source graph under frozen W transfer. No new bank query, geometry edit, or physical admission.',
 'input_pins':{
  'baseline103_edge_report_sha256':edgehash,'compact084_mesh_sha256':compacthash,
  'Wnew_detail_to_084_sha256':transferhash,'detail_NHSKIN_sha256':detailhash,
  'accepted_transfer_rows_sha256':rowhash,'accepted_coarse_faces_sha256':sha(COARSE_FACES),
  'facet_marker_map_sha256':sha(MARKER_MAP),'baseline103_candidate_sha256':sha(BASE/'candidate.npz'),
  'initial_diagnosis_script_sha256':sha(SCRIPT1),'initial_diagnosis_report_sha256':sha(REPORT1)},
 'edge':{'compact_output_rows':list(TARGET_EDGE),'source_global_node_ids':[int(globalids[TARGET_EDGE[0]]),int(globalids[TARGET_EDGE[1]])],
  'source_input_rows':[int(srcrows[i]) for i in TARGET_EDGE],
  'baseline_current_world_chord_length_mm':coarse_current_chord,
  'reference_coarse_chord_length_mm':float(np.linalg.norm(ref[TARGET_EDGE[0]]-ref[TARGET_EDGE[1]])*1000),
  'baseline103_exact_bank_hit_witnesses':edge_witnesses,
  'target_bank':body_map[66],
  'interpretation':'This is a compact O-surface edge, not an interior tetrahedral chord. Its exact baseline segment crosses the recorded body66/stable100 bank triangles. The same endpoint pair is joined through a longer, laterally displaced path in the detailed NHSKIN graph.'},
 'outer_surface_parentage':{'output_O_faces':output_faces,'source_O_face_rows':source_face_rows_for_edge,
   'source_O_face_vertices':{str(i):list(map(int,source_o_faces[i])) for i in parent_rows},
   'accepted_coarse_face_vertices':{str(i):coarse_face_rows[i] for i in parent_rows},
   'marker_to_source_face_parent_verified':True},
 'detailed_surface_patch':{
   'detail_header':detail_header,'parent_face_rows':parent_rows,
   'active_detail_vertices_selected_by_parent_face':{str(k):int(v) for k,v in per_parent.items()},
   'selected_detail_vertex_count':len(patch_vertices),'selected_detail_face_edge_count':len(edge_set),
   'direct_anchor_detail_rows':{str(TARGET_EDGE[0]):start,str(TARGET_EDGE[1]):goal},
   'direct_anchor_donors':{str(TARGET_EDGE[0]):{'detail_donor_body':int(transfer_rows[start]['detail_donor_body']),'transfer_donor_body':int(transfer_rows[start]['transfer_donor_body'])},
                           str(TARGET_EDGE[1]):{'detail_donor_body':int(transfer_rows[goal]['detail_donor_body']),'transfer_donor_body':int(transfer_rows[goal]['transfer_donor_body'])}},
   'W_reference_to_accepted_detail_origin':{'active_rows':int(np.count_nonzero(active)),'max_residual_m':float(map_residual.max()),'p99_residual_m':float(np.quantile(map_residual,.99))},
   'shortest_surface_edge_path':{'vertex_count':len(path),'edge_count':len(path)-1,
       'length_mm':float(dist[target_local]*1000),'endpoint_chord_mm':float(chord_len*1000),
       'path_to_chord_ratio':float(dist[target_local]/chord_len),
       'max_lateral_distance_to_endpoint_chord_mm':float(lateral_mm.max()),
       'p95_lateral_distance_to_endpoint_chord_mm':float(np.quantile(lateral_mm,.95)),
       'minimum_chord_parameter':float(projected_t.min()),'maximum_chord_parameter':float(projected_t.max()),
       'signed_side_extent_relative_to_parent_face0_mm':[float(signed_side_mm.min()),float(signed_side_mm.max())],
       'path_vertices_on_both_sides_of_parent_face0_chord_plane':bool(signed_side_mm.min() < -1e-3 and signed_side_mm.max() > 1e-3),
       'selected_parent_face_rows_on_path':{str(k):int(v) for k,v in path_face_counts.items()},
       'detail_donor_bodies_on_path':{str(k):int(v) for k,v in path_detail_donors.items()},
       'vertices':path_data},
   'coarse_chord_hit_point_to_detailed_path_mm':hit_to_path,
   'interpretation_limit':'W maps each detailed row to coarse source faces; the path is an exact NHSKIN face-edge path whose endpoints are direct W one-hot anchors. This diagnoses a coarse boundary shortcut versus a fine connected route, but does not certify that the detailed path clears all triangles of the target bone bank.'},
 'target_digit_bank_baseline103_edge_groups':target_bank_summary,
 'limitations':['No new exact test was run on the detailed path against the complete 185-bank triangle set.',
  'The reported hit-to-path distance is geometric proximity only, not clearance.',
  'The path and source-face ancestry are construction evidence; they do not qualify a surface refinement or claim a physical cause.']}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(OUT),'edge':report['edge'],'outer_surface_parentage':report['outer_surface_parentage'],
 'shortest_path':{k:v for k,v in report['detailed_surface_patch']['shortest_surface_edge_path'].items() if k!='vertices'},
 'target_bank_baseline103_edge_groups':target_bank_summary,'hit_to_path':hit_to_path},indent=2))
