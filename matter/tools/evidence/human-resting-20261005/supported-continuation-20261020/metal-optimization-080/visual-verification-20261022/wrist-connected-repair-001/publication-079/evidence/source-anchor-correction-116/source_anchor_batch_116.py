#!/Users/n/numi-human-prep-venv-20261005/bin/python
from __future__ import annotations
import csv,gzip,json,hashlib,importlib.util,math,sys,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

WORK=Path('/Users/n/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=WORK/'digit-source-diagnosis-110'
OUTDIR=D/'source-anchor-correction-116'
OUTDIR.mkdir(parents=True,exist_ok=True)
OLD=D/'batch-146-curved-source-midpoints-quality-114.json'
COARSE=WORK/'publication-071/evidence/exact-transfer-075/run-001/source-preflight/accepted-coarse-reference.csv.gz'
ROWS=E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv'
FACES_CSV=E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-coarse-faces.csv'
DETAIL=E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'
COMPACT=WORK/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
TRANSFER=WORK/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz'
MARKER_MAP=WORK/'whole-reference-tetgen-084-bounded/facet-marker-map.csv'
OUT=OUTDIR/'batch-146-corrected-source-anchors.json'
COLUMN_COUNT=2411
DIRECT_SENTINEL=4294967295
ACTIVE_EPS=1e-10
TIE_TOL=1e-12

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def detail_anchor(coarse_record, row_by_source_id, detail_xyz, rows):
 """Resolve geometric anchor from accepted source provenance, never W support."""
 sid=int(coarse_record['source_direct_id'])
 if sid != DIRECT_SENTINEL:
  if sid not in row_by_source_id: raise ValueError(f'non-sentinel source_direct_id {sid} is absent')
  dr=int(row_by_source_id[sid]); p=np.asarray(detail_xyz[dr],dtype=np.float64)
  coarse_world=np.asarray([float(coarse_record[f'world_{a}_m']) for a in 'xyz'])
  face_ids=[int(coarse_record[f'source_face_v{i}']) for i in range(3)]
  face_bary=np.asarray([float(coarse_record[f'source_bary{i}']) for i in range(3)],dtype=np.float64)
  face_res=None
  if len(set(face_ids))==3 and all(q in row_by_source_id for q in face_ids) and np.isfinite(face_bary).all() and abs(float(face_bary.sum())-1.0)<=1e-10:
   face_point=face_bary@np.asarray(detail_xyz[[row_by_source_id[q] for q in face_ids]],dtype=np.float64)
   face_res=float(np.linalg.norm(p-face_point))
  return {'kind':'direct_source_vertex','source_direct_id':sid,'detail_row':dr,'detail_source_vertex_id':int(rows[dr]['detail_source_vertex_id']),
    'point_m':p,'source_face_vertex_ids':face_ids,
    'barycentric':face_bary.tolist(),'direct_vs_source_face_bary_residual_m':face_res,
    'coarse_world_m':coarse_world,'coarse_world_residual_m':float(np.linalg.norm(p-coarse_world)),
    'vertex_options':[(dr,0.0)]}
 vids=[int(coarse_record[f'source_face_v{i}']) for i in range(3)]
 bary=np.asarray([float(coarse_record[f'source_bary{i}']) for i in range(3)],dtype=np.float64)
 if len(set(vids))!=3 or any(v not in row_by_source_id for v in vids): raise ValueError('invalid fallback source face IDs')
 if not np.isfinite(bary).all() or abs(float(bary.sum())-1)>1e-10 or float(np.min(bary)) < -1e-10: raise ValueError('invalid fallback barycentric weights')
 drs=[int(row_by_source_id[v]) for v in vids]
 p=bary@np.asarray(detail_xyz[drs],dtype=np.float64)
 cw=np.asarray([float(coarse_record[f'world_{a}_m']) for a in 'xyz'])
 return {'kind':'source_face_barycentric_virtual_point','source_direct_id':None,'detail_row':None,'detail_source_vertex_id':None,
   'point_m':p,'source_face_vertex_ids':vids,'source_face_detail_rows':drs,'barycentric':bary,
   'coarse_world_m':cw,'coarse_world_residual_m':float(np.linalg.norm(p-cw)),
   'vertex_options':[(int(dr),float(np.linalg.norm(detail_xyz[dr]-p))) for dr in drs]}

def interp_dict(a,b,t):
 out=defaultdict(float)
 for k,v in a.items(): out[int(k)]+=(1-float(t))*float(v)
 for k,v in b.items(): out[int(k)]+=float(t)*float(v)
 return dict(out)

def row_w_map(W,dr):
 lo,hi=W.indptr[int(dr):int(dr)+2]
 return {int(j):float(w) for j,w in zip(W.indices[lo:hi],W.data[lo:hi]) if abs(float(w))>ACTIVE_EPS}

def anchor_w_map(anchor,W,row_by_source_id):
 if anchor['kind']=='direct_source_vertex': return row_w_map(W,anchor['detail_row'])
 out=defaultdict(float)
 for sid,bw in zip(anchor['source_face_vertex_ids'],anchor['barycentric']):
  for col,w in row_w_map(W,row_by_source_id[int(sid)]).items():out[col]+=float(bw)*w
 return dict(out)

def main():
 t0=time.monotonic()
 b=loadmod('source_batch_110',D/'source_path_batch_110_v2.py')
 o=loadmod('orientation_112',D/'orientation_scalar_audit_112.py')
 pins={'old_batch':sha(OLD),'accepted_coarse_reference_gz':sha(COARSE),'accepted_transfer_rows':sha(ROWS),'accepted_coarse_faces':sha(FACES_CSV),
   'nhskin':sha(DETAIL),'compact_candidate':sha(COMPACT),'Wnew':sha(TRANSFER),'runner_sha256':sha(Path(__file__))}
 old=json.loads(OLD.read_text())
 with np.load(COMPACT,allow_pickle=False) as z:
  R=np.asarray(z['reference_points_m'],dtype=np.float64); C=np.asarray(z['current037_points_m_f32']); srcrows=np.asarray(z['source_input_row'],dtype=np.int64)
  gids=np.asarray(z['output_node_ids'],dtype=np.int64); source_o_faces=np.asarray(z['source_O_faces'],dtype=np.int64)
 with np.load(TRANSFER,allow_pickle=False) as z: W=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(int(x) for x in z['shape']))
 legacy_onehot=defaultdict(list)
 for dr in range(W.shape[0]):
  lo,hi=W.indptr[dr:dr+2]; nz=[(int(j),float(w)) for j,w in zip(W.indices[lo:hi],W.data[lo:hi]) if abs(float(w))>ACTIVE_EPS]
  if len(nz)==1 and abs(nz[0][1]-1.0)<=1e-7: legacy_onehot[nz[0][0]].append(dr)
 faces,header=b.parse_detail(DETAIL)
 rows=list(csv.DictReader(ROWS.open(newline='')))
 with gzip.open(COARSE,'rt',newline='') as f: coarse=list(csv.DictReader(f))
 if len(coarse)!=COLUMN_COUNT or len(rows)!=header['vertex_count'] or W.shape[0]!=len(rows):raise RuntimeError('input row domains disagree')
 detail_xyz=np.asarray([[float(r[f'detail_rest_world_{a}_m']) for a in 'xyz'] for r in rows],dtype=np.float64)
 active=np.asarray([int(r['active']) for r in rows],bool)
 selected=np.asarray([int(r['selected_face_row']) for r in rows],dtype=np.int64)
 donor=np.asarray([int(r['detail_donor_body']) for r in rows],dtype=np.int64)
 tdonor=np.asarray([int(r['transfer_donor_body']) for r in rows],dtype=np.int64)
 source_id_to_rows=defaultdict(list)
 for i,r in enumerate(rows):source_id_to_rows[int(r['detail_source_vertex_id'])].append(i)
 if any(len(v)!=1 for v in source_id_to_rows.values()):raise RuntimeError('source vertex IDs are not unique')
 row_by_id={k:v[0] for k,v in source_id_to_rows.items()}
 anchors=[]
 for i,r in enumerate(coarse):
  if int(r['coarse_row'])!=i:raise RuntimeError('coarse reference row order mismatch')
  a=detail_anchor(r,row_by_id,detail_xyz,rows)
  if a['kind']=='direct_source_vertex' and not np.isfinite(a['coarse_world_residual_m']):raise RuntimeError('non-finite direct source anchor residual')
  anchors.append(a)
 # Exact detailed face and coarse face lookups.
 edge_to_coarse_faces=defaultdict(list)
 for fr,tri in enumerate(source_o_faces):
  for a,b0 in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):edge_to_coarse_faces[tuple(sorted((int(a),int(b0))))].append(fr)
 coarse_face_allowed=defaultdict(set)
 for i,r in enumerate(rows):
  if active[i] and 0<=selected[i]<len(source_o_faces):coarse_face_allowed[int(selected[i])].add(i)
 detail_faces_by_set=defaultdict(list)
 for fi,tri in enumerate(faces):detail_faces_by_set[tuple(sorted(map(int,tri)))].append(fi)
 id_to_faces=defaultdict(list)
 for fi,tri in enumerate(faces):
  for sid in tri:id_to_faces[int(rows[int(sid)]['detail_source_vertex_id'])].append(fi)
 detail_edge_faces=defaultdict(list)
 for fi,tri in enumerate(faces):
  a,b0,c=map(int,tri)
  for x,y in ((a,b0),(b0,c),(c,a)):detail_edge_faces[tuple(sorted((x,y)))].append(fi)
 # Import a verified complete transfer displacement map; use W only for displacement, never geometric anchoring.
 detail_current=detail_xyz+W.dot(C.astype(np.float64)-R)
 flips,components,top=o.orient_faces_by_edge_adjacency(faces)
 fv_r,ff_r,fc_r=o.area_normals(detail_xyz,faces,flips)
 fv_c,ff_c,fc_c=o.area_normals(detail_current,faces,flips)
 # Outward sign is calibrated only from exact direct source IDs and the O->I source layer direction.
 signs=[]
 for cr,a in enumerate(anchors):
  if a['kind']!='direct_source_vertex':continue
  dr=a['detail_row']; d=R[cr]-R[cr+COLUMN_COUNT]
  q=float(np.dot(fv_r[dr],d))
  if abs(q)>1e-14:signs.append(q)
 if not signs:raise RuntimeError('no stable direct anchor for global outward orientation')
 outward_sign=1 if float(np.median(signs))>0 else -1
 if sum(x>0 for x in signs) < .9*len(signs) and sum(x<0 for x in signs) < .9*len(signs):raise RuntimeError('direct-source outward orientation is not coherent')
 # Use all unique edges already enumerated in frozen 114 source proposal.
 edges=[tuple(map(int,e['edge_output_rows'])) for e in old['edges']]
 if len(edges)!=146 or len(set(tuple(sorted(e)) for e in edges))!=146:raise RuntimeError('expected 146 unique proposal edges')
 # Parse exact source O faces.
 src_edges_to_faces=edge_to_coarse_faces
 output=[]; anchor_deltas=[]; legacy_min_deltas=[]; status=Counter()
 for ei,(u,v) in enumerate(edges):
  endpoint_anchors=[anchors[u],anchors[v]]
  oldrec=old['edges'][ei]
  parent=edge_to_coarse_faces.get(tuple(sorted((u,v))),[])
  if len(parent)!=2: raise RuntimeError(f'edge {(u,v)} has {len(parent)} source O parent faces')
  # Anchor candidate vertex rows carry path connector cost from exact provenance point.
  endpoint_options=[a['vertex_options'] for a in endpoint_anchors]
  endpoint_donors=set()
  for a,opts in zip(endpoint_anchors,endpoint_options):
   rr=[]
   if a['detail_row'] is not None:rr=[int(a['detail_row'])]
   else:rr=list(a['source_face_detail_rows'])
   for dr in rr:endpoint_donors.update((int(donor[dr]),int(tdonor[dr])))
  parent_rows=set()
  for fr in parent:parent_rows.update(coarse_face_allowed.get(int(fr),set()))
  allowed=set(endpoint_donors)
  for dr in parent_rows:allowed.update((int(donor[dr]),int(tdonor[dr])))
  graph=defaultdict(list)
  for (aa,bb),fis in detail_edge_faces.items():
   if not active[aa] or not active[bb]:continue
   if int(donor[aa]) not in allowed or int(donor[bb]) not in allowed or int(tdonor[aa]) not in allowed or int(tdonor[bb]) not in allowed:continue
   length=float(np.linalg.norm(detail_xyz[aa]-detail_xyz[bb]))
   if length>0:graph[aa].append((bb,length));graph[bb].append((aa,length))
  choices=[]
  for sr,scost in endpoint_options[0]:
   for gr,gcost in endpoint_options[1]:
    if int(sr)==int(gr):
     path=[int(sr)];info={'connected':True,'distance_m':0.0,'shortest_path_count_capped_2':1,'tie_tolerance_m':TIE_TOL}
    else:path,info=b.shortest_path_counted(graph,int(sr),int(gr),TIE_TOL)
    if path is None:continue
    total=float(scost)+float(info['distance_m'])+float(gcost)
    sidseq=tuple(int(rows[x]['detail_source_vertex_id']) for x in path)
    choices.append((total,sidseq,int(sr),int(gr),path,info,float(scost),float(gcost)))
  if not choices:raise RuntimeError(f'no corrected source path for edge {(u,v)}')
  choices.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
  best=choices[0];_,_,start,goal,path,pathinfo,startcost,endcost=best
  # Build an exact polyline from virtual provenance anchors through source vertices to target anchor.
  poly=[]
  def push(point,support,bary,kind,detail_row=None,source_face_rows=None):
   point=np.asarray(point,dtype=np.float64)
   if poly and np.linalg.norm(point-poly[-1]['point'])<=1e-14:
    # Coincident provenance/detail points must carry the same displacement trace.
    keys=set(poly[-1]['W'])|set(support)
    if max((abs(poly[-1]['W'].get(k,0.)-support.get(k,0.)) for k in keys),default=0.)>1e-7:raise RuntimeError('coincident source anchors disagree in W displacement trace')
    return
   poly.append({'point':point,'W':dict(support),'bary':dict(bary),'kind':kind,'detail_row':detail_row,'source_face_rows':list(source_face_rows or [])})
  def anchor_bary(a):
   if a['kind']=='direct_source_vertex':return {int(a['detail_row']):1.0}
   return {int(row_by_id[sid]):float(w) for sid,w in zip(a['source_face_vertex_ids'],a['barycentric'])}
  # source anchor point then path vertices; for a direct anchor the first is coincident and collapsed.
  for endpoint_i,vertex in ((0,start),(1,goal)):
   pass
  a0,a1=endpoint_anchors
  push(a0['point_m'],anchor_w_map(a0,W,row_by_id),anchor_bary(a0),a0['kind'],a0['detail_row'],
       [next(iter(detail_faces_by_set[tuple(sorted(a0['source_face_detail_rows']))]))] if a0['kind']!='direct_source_vertex' and len(detail_faces_by_set[tuple(sorted(a0['source_face_detail_rows']))])==1 else [])
  for dr in path:
   push(detail_xyz[dr],row_w_map(W,dr),{int(dr):1.0},'detail_vertex',int(dr),[])
  push(a1['point_m'],anchor_w_map(a1,W,row_by_id),anchor_bary(a1),a1['kind'],a1['detail_row'],
       [next(iter(detail_faces_by_set[tuple(sorted(a1['source_face_detail_rows']))]))] if a1['kind']!='direct_source_vertex' and len(detail_faces_by_set[tuple(sorted(a1['source_face_detail_rows']))])==1 else [])
  if len(poly)<2:raise RuntimeError('path polyline collapsed to one point')
  pts=np.asarray([x['point'] for x in poly],dtype=np.float64)
  lengths=np.linalg.norm(np.diff(pts,axis=0),axis=1)
  if np.any(lengths<=1e-15):raise RuntimeError(f'zero-length corrected source path segment for {(u,v)}')
  cumulative=np.r_[0.,np.cumsum(lengths)];total=float(cumulative[-1]);target=.5*total
  seg=int(min(len(lengths)-1,max(0,np.searchsorted(cumulative,target,side='right')-1)))
  alpha=float((target-cumulative[seg])/lengths[seg]);p=(1-alpha)*pts[seg]+alpha*pts[seg+1]
  wmid=interp_dict(poly[seg]['W'],poly[seg+1]['W'],alpha)
  delta=sum((w*(C[int(k)].astype(np.float64)-R[int(k)]) for k,w in wmid.items()),start=np.zeros(3))
  pc=p+delta
  # Select exact source faces containing the active path segment.
  rec0,rec1=poly[seg],poly[seg+1]
  incident=[]
  if rec0['detail_row'] is not None and rec1['detail_row'] is not None:
   incident=detail_edge_faces.get(tuple(sorted((int(rec0['detail_row']),int(rec1['detail_row'])))),[])
  if not incident:incident=sorted(set(rec0['source_face_rows']+rec1['source_face_rows']))
  if not incident:
   # Determine triangle support from the source interpolation rows; fail closed if not one exact NHSKIN face.
   common=set(rec0['bary'])|set(rec1['bary'])
   incident=[fi for fi,tri in enumerate(faces) if set(map(int,tri)).issubset(common) and len(common)<=3]
  if not incident:raise RuntimeError(f'no exact NHSKIN face support for source segment of {(u,v)}')
  nr=np.sum(fc_r[incident],axis=0);nc=np.sum(fc_c[incident],axis=0)
  nrn=float(np.linalg.norm(nr));ncn=float(np.linalg.norm(nc))
  if nrn<=1e-30 or ncn<=1e-30:raise RuntimeError('degenerate incident source normal')
  nr=outward_sign*nr/nrn;nc=outward_sign*nc/ncn
  # Existing scalar skin/fat thickness fields at exact coarse edge endpoints; no fixed thickness added.
  sr=[];fr=[];sc=[];fc=[]
  for cr in (u,v):
   sr.append(float(-np.dot(R[cr+COLUMN_COUNT]-R[cr],nr)))
   fr.append(float(-np.dot(R[cr+2*COLUMN_COUNT]-R[cr+COLUMN_COUNT],nr)))
   sc.append(float(-np.dot(C[cr+COLUMN_COUNT].astype(float)-C[cr].astype(float),nc)))
   fc.append(float(-np.dot(C[cr+2*COLUMN_COUNT].astype(float)-C[cr+COLUMN_COUNT].astype(float),nc)))
  skin_r=float(np.mean(sr));fat_r=float(np.mean(fr));skin_c=float(np.mean(sc));fat_c=float(np.mean(fc))
  prop_r={'O_m':p.tolist(),'I_m':(p-skin_r*nr).tolist(),'D_m':(p-(skin_r+fat_r)*nr).tolist(),'outward_normal':nr.tolist(),'skin_depth_m':skin_r,'fat_depth_m':fat_r,'depths_are_interpolated_existing_scalar_fields':True}
  prop_c={'O_m':pc.tolist(),'I_m':(pc-skin_c*nc).tolist(),'D_m':(pc-(skin_c+fat_c)*nc).tolist(),'outward_normal':nc.tolist(),'skin_depth_m':skin_c,'fat_depth_m':fat_c,'depths_are_interpolated_existing_scalar_fields':True}
  oldpath=oldrec.get('path_detail_rows',[])
  old_endpoint_rows=[int(oldpath[0]) if oldpath else None,int(oldpath[-1]) if oldpath else None]
  corrected_endpoint_records=[]
  for ei2,(cr,a,oldrow) in enumerate(zip((u,v),endpoint_anchors,old_endpoint_rows)):
   oldpoint=detail_xyz[oldrow] if oldrow is not None else None
   wopts=sorted(legacy_onehot.get(cr,[]))
   minrow=wopts[0] if wopts else None
   minpoint=detail_xyz[minrow] if minrow is not None else None
   min_dist=float(np.linalg.norm(minpoint-a['point_m'])) if minpoint is not None else None
   if min_dist is not None: legacy_min_deltas.append(min_dist)
   corrected_endpoint_records.append({'coarse_output_row':cr,'method':a['kind'],'source_direct_id':a['source_direct_id'],
      'detail_row':a['detail_row'],'source_face_vertex_ids':a['source_face_vertex_ids'],'source_face_detail_rows':a.get('source_face_detail_rows'),
      'barycentric':np.asarray(a['barycentric']).tolist(),'authoritative_anchor_m':a['point_m'].tolist(),
      'coarse_reference_world_m':a['coarse_world_m'].tolist(),'coarse_to_anchor_residual_m':a['coarse_world_residual_m'],
      'old_W_onehot_path_endpoint_detail_row':oldrow,'old_W_onehot_endpoint_m':oldpoint.tolist() if oldpoint is not None else None,
      'old_to_authoritative_anchor_distance_m':float(np.linalg.norm(oldpoint-a['point_m'])) if oldpoint is not None else None,
      'W_onehot_candidate_count':len(wopts),'legacy_min_detail_row_onehot':minrow,
      'legacy_min_detail_source_vertex_id':int(rows[minrow]['detail_source_vertex_id']) if minrow is not None else None,
      'legacy_min_onehot_to_authoritative_anchor_distance_m':min_dist})
   if oldpoint is not None:anchor_deltas.append(float(np.linalg.norm(oldpoint-a['point_m'])))
  output.append({'edge_output_rows':[u,v],'edge_global_node_ids':[int(gids[u]),int(gids[v])],
    'source_kind':'accepted_coarse_reference_direct_or_exact_face_barycentric_anchor_geodesic',
    'status':'SOURCE_PATH_PROPOSAL_DIAGNOSTIC_ONLY','proposal_eligible':True,
    'crossing_banks':oldrec.get('crossing_banks',[]),'quality_witness':oldrec.get('quality_witness'),
    'parent_source_O_face_rows':parent,'anchor_method_per_endpoint':[a['kind'] for a in endpoint_anchors],
    'corrected_endpoint_anchor_records':corrected_endpoint_records,
    'allowed_local_donor_ids':sorted(allowed),
    'old_path_detail_rows':[int(x) for x in oldpath],
    'old_path_same_detail_row_sequence':bool([int(x) for x in oldpath]==[int(x) for x in path]),
    'path_detail_rows':[int(x) for x in path],
    'path_source_vertex_ids':[int(rows[x]['detail_source_vertex_id']) for x in path],
    'path_length_m':total,'shortest_path_connector_lengths_m':[startcost,endcost],
    'shortest_path':{'count_capped_2':int(pathinfo.get('shortest_path_count_capped_2',1)),'tie_tolerance_m':TIE_TOL,
       'deterministic_tiebreak':'shortest total including exact virtual-anchor connectors, then lexicographic source vertex ID sequence'},
    'midpoint_polyline_segment_index':seg,'midpoint_segment_fraction':alpha,'source_midpoint_reference_m':p.tolist(),'source_midpoint_current037_m':pc.tolist(),
    'midpoint_detail_face_rows':list(map(int,incident)),
    'midpoint_source_barycentric_detail_rows':[{'detail_row':int(k),'weight':float(w)} for k,w in sorted(interp_dict(poly[seg]['bary'],poly[seg+1]['bary'],alpha).items())],
    'midpoint_W_displacement_support_rows':sorted(wmid),'midpoint_W_displacement_support_weights':[float(wmid[k]) for k in sorted(wmid)],
    'outward_normal_sign_from_exact_direct_source_layer_order':outward_sign,
    'reference_outward_normal':nr.tolist(),'current037_outward_normal':nc.tolist(),
    'source_skin_depth_reference_endpoint_m':sr,'source_fat_depth_reference_endpoint_m':fr,
    'source_skin_depth_current037_endpoint_m':sc,'source_fat_depth_current037_endpoint_m':fc,
    'normal_scalar_source_proposal_reference':prop_r,'normal_scalar_source_proposal_current037':prop_c,
    'limitations':['Source anchors use accepted-coarse-reference direct IDs or exact face+barycentric virtual points; W is only used to prolong displacement.','Midpoint paths and normal-offset layers are construction proposals; no new whole-mesh self/bone/material/mass gate was run.','Any signed negative scalar depth is retained as a source witness and is not abs/clamped.']})
  status['direct_direct' if all(a['kind']=='direct_source_vertex' for a in endpoint_anchors) else 'contains_virtual_anchor']+=1
 # Diagnostics summary old anchor errors.
 direct_res=[float(a['coarse_world_residual_m']) for a in anchors if a['kind']=='direct_source_vertex']
 direct_face_res=[float(a['direct_vs_source_face_bary_residual_m']) for a in anchors if a['kind']=='direct_source_vertex' and a['direct_vs_source_face_bary_residual_m'] is not None]
 virtual_res=[float(a['coarse_world_residual_m']) for a in anchors if a['kind']!='direct_source_vertex']
 result={'status':'CORRECTED_146_ANCHOR_SOURCE_PATHS_PROPOSAL_ONLY','scope':'Rebuilds all frozen 146 O-edge paths from accepted coarse-reference source provenance. W is never used to choose geometric anchors; only midpoint displacement support is prolonged through Wnew. No mesh/geometry asset is modified.',
  'input_pins':pins,'counts':{'edge_records':len(output),'direct_source_rows':sum(a['kind']=='direct_source_vertex' for a in anchors),'virtual_sourceface_rows':sum(a['kind']!='direct_source_vertex' for a in anchors),
    'corrected_direct_endpoints':sum(a['kind']=='direct_source_vertex' for e in output for a in [anchors[int(e['edge_output_rows'][0])],anchors[int(e['edge_output_rows'][1])]]),
    'corrected_virtual_endpoints':sum(a['kind']!='direct_source_vertex' for e in output for a in [anchors[int(e['edge_output_rows'][0])],anchors[int(e['edge_output_rows'][1])]]),
    'anchor_modes_by_edge':dict(status),'corrected_path_sequences_differ_from_old':sum(not e['old_path_same_detail_row_sequence'] for e in output),'edges_with_shortest_path_ties':sum(e['shortest_path']['count_capped_2']>1 for e in output)},
  'anchor_audit':{'direct_source_coordinate_residual_m':{'count':len(direct_res),'max':max(direct_res),'median':float(np.median(direct_res))},
    'direct_id_vs_source_face_bary_residual_m':{'count':len(direct_face_res),'max':max(direct_face_res) if direct_face_res else None,'median':float(np.median(direct_face_res)) if direct_face_res else None},
    'virtual_anchor_vs_coarse_world_residual_m':{'count':len(virtual_res),'max':max(virtual_res) if virtual_res else None,'median':float(np.median(virtual_res)) if virtual_res else None},
    'old_selected_path_endpoint_to_authoritative_anchor_distance_m':{'count':len(anchor_deltas),'max':max(anchor_deltas),'median':float(np.median(anchor_deltas)),'p95':float(np.percentile(anchor_deltas,95))},
    'legacy_min_W_onehot_row_to_authoritative_anchor_distance_m':{'count':len(legacy_min_deltas),'max':max(legacy_min_deltas),'median':float(np.median(legacy_min_deltas)),'p95':float(np.percentile(legacy_min_deltas,95)),'over_1mm':sum(x>.001 for x in legacy_min_deltas)},
    'largest_discrepancies_m':sorted([{'edge_output_rows':e['edge_output_rows'],'endpoint':q['coarse_output_row'],'old_detail_row':q['old_W_onehot_path_endpoint_detail_row'],'distance_m':q['old_to_authoritative_anchor_distance_m']} for e in output for q in e['corrected_endpoint_anchor_records'] if q['old_to_authoritative_anchor_distance_m'] is not None],key=lambda x:x['distance_m'],reverse=True)[:20],
    'largest_legacy_min_W_onehot_discrepancies_m':sorted([{'edge_output_rows':e['edge_output_rows'],'endpoint':q['coarse_output_row'],'source_direct_id':q['source_direct_id'],'legacy_min_detail_source_vertex_id':q['legacy_min_detail_source_vertex_id'],'distance_m':q['legacy_min_onehot_to_authoritative_anchor_distance_m']} for e in output for q in e['corrected_endpoint_anchor_records'] if q['legacy_min_onehot_to_authoritative_anchor_distance_m'] is not None],key=lambda x:x['distance_m'],reverse=True)[:20]},
  'orientation':{'method':'oriented NHSKIN adjacency faces; global outward sign selected by median dot of detailed vertex normals against exact direct-source O-minus-I vectors','sign':outward_sign,'direct_anchor_samples':len(signs),'positive_after_sign_fraction':float(np.mean(np.asarray(signs)*outward_sign>0))},
  'edges':output,'limitations':['This corrects geometric source-anchor provenance; it does not establish the old source path is anatomically optimal or collision-free.','All 146 records remain proposals only. Exact full-surface, connected-volume, bone clearance, layer thickness, mass, and current-pose gates remain required.','Direct anchor IDs are source provenance identifiers, not W row argmaxes. Fallback points are exact accepted source-face barycentric points, with source-point versus coarse world residual retained.'],
  'elapsed_seconds':time.monotonic()-t0}
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'output':str(OUT),'sha256':sha(OUT),'counts':result['counts'],'anchor_audit':result['anchor_audit'],'orientation':result['orientation'],'elapsed_seconds':result['elapsed_seconds']},indent=2))

if __name__=='__main__':main()
