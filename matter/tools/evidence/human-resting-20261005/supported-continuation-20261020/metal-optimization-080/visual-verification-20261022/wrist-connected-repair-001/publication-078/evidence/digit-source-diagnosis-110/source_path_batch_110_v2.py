#!/Users/n/numi-human-prep-venv-20261005/bin/python
"""Bounded source-supported midpoint proposal for baseline O-O crossing edges."""
from __future__ import annotations
import csv,hashlib,itertools,json,struct,heapq,math,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

WORK=Path('/Users/n/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=WORK/'digit-source-diagnosis-110'
BASE=WORK/'connected-cell-witness-103/run-001/baseline'
COMPACT=WORK/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
TRANSFER=WORK/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002'
EDGE_REPORT=BASE/'tet-edge-report.json'
DETAIL=E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'
ROWS=E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv'
FACES_CSV=E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-coarse-faces.csv'
MARKER_MAP=WORK/'whole-reference-tetgen-084-bounded/facet-marker-map.csv'
OUT=D/'batch-145-curved-source-midpoints-004.json'
COLUMN_COUNT=2411
SUPPORT_TOL=1e-10
SHORTEST_TIE_TOL_M=1e-12
EXPECTED={
 'compact':'5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
 'edge_report':'41c8b9625e99ae6f5a4b4f3d909e847c906c53adc7523b221e0f2373faa7c2ee',
 'transfer':'55329313bf9c2c3b613fcd7eb52dd9b7b453520dd1b4a14e0e0d6ac457ce8519',
 'detail':'7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb',
 'rows':'71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14',
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def path_midpoint(points):
 pts=np.asarray(points,dtype=np.float64)
 if pts.ndim!=2 or pts.shape[1]!=3 or len(pts)<2 or not np.isfinite(pts).all():
  raise ValueError('path must be finite Nx3 with N>=2')
 seg=np.linalg.norm(np.diff(pts,axis=0),axis=1)
 if np.any(seg<=0):raise ValueError('path has zero-length edge')
 cumulative=np.r_[0.0,np.cumsum(seg)];total=float(cumulative[-1]);target=.5*total
 i=min(len(seg)-1,max(0,int(np.searchsorted(cumulative,target,side='right')-1)))
 alpha=float((target-cumulative[i])/seg[i])
 p=(1-alpha)*pts[i]+alpha*pts[i+1]
 return {'total_length_m':total,'fraction':.5,'edge_index':i,'edge_fraction':alpha,'point_m':p,'segment_lengths_m':seg,'cumulative_m':cumulative}

def shortest_path_counted(graph,start,goal,tie_tol=SHORTEST_TIE_TOL_M):
 """Stable row-ID Dijkstra plus capped count of paths within tie tolerance."""
 dist={int(start):0.0};prev={int(start):None};heap=[(0.0,int(start))]
 while heap:
  du,u=heapq.heappop(heap)
  if du>dist.get(u,float('inf'))+tie_tol:continue
  for v,w in sorted(graph.get(u,())):
   alt=du+w;old=dist.get(v,float('inf'))
   if alt<old-tie_tol:
    dist[v]=alt;prev[v]=u;heapq.heappush(heap,(alt,v))
   elif abs(alt-old)<=tie_tol and v!=start:
    if prev.get(v) is None or u<prev[v]:prev[v]=u
 if goal not in dist:return None,{'connected':False,'shortest_path_count_capped_2':0,'distance_m':None}
 order=sorted(dist,key=lambda n:(dist[n],n));ways={n:0 for n in order};ways[start]=1
 for u in order:
  if not ways[u]:continue
  for v,w in graph.get(u,()):
   if dist[v]>dist[u]+tie_tol and abs((dist[u]+w)-dist[v])<=tie_tol:
    ways[v]=min(2,ways[v]+ways[u])
 path=[];cur=int(goal)
 while cur is not None:
  path.append(cur)
  if cur==start:break
  cur=prev.get(cur)
 if not path or path[-1]!=start:return None,{'connected':False,'shortest_path_count_capped_2':int(ways.get(goal,0)),'distance_m':float(dist[goal])}
 return path[::-1],{'connected':True,'shortest_path_count_capped_2':int(ways.get(goal,0)),'distance_m':float(dist[goal]),'tie_tolerance_m':tie_tol}

def edge_midpoint_field(ref,current,u,v,fraction,column_count=COLUMN_COUNT):
 t=float(fraction)
 if not 0<=t<=1:raise ValueError('fraction outside [0,1]')
 out={}
 for layer,offset in (('O',0),('I',column_count),('D',2*column_count)):
  a=int(u)+offset;b=int(v)+offset
  out[layer]={'endpoint_rows':[a,b], 'reference_endpoint_m':[ref[a].tolist(),ref[b].tolist()],
              'current_endpoint_m':[current[a].astype(float).tolist(),current[b].astype(float).tolist()],
              'reference_interpolated_m':((1-t)*ref[a]+t*ref[b]).tolist(),
              'current_interpolated_m':((1-t)*current[a].astype(float)+t*current[b].astype(float)).tolist()}
 return out

def construct_midpoint_layers(midpoint_reference_m, midpoint_current037_m, reference_points_m, current037_points_m_f32, endpoint_rows, fraction, column_count=COLUMN_COUNT):
    """Place I/D by transporting existing endpoint offsets from a source-path O midpoint."""
    t=float(fraction)
    if not 0.0 <= t <= 1.0: raise ValueError('fraction outside [0,1]')
    u,v=map(int,endpoint_rows)
    ref=np.asarray(reference_points_m,dtype=np.float64)
    cur=np.asarray(current037_points_m_f32)
    out={}
    for layer,offset in (('O',0),('I',column_count),('D',2*column_count)):
        ou,ov=u,v; iu,iv=ou+offset,ov+offset
        if iv>=len(ref) or iv>=len(cur): raise ValueError('endpoint layer row out of bounds')
        ro_u=ref[iu]-ref[ou]; ro_v=ref[iv]-ref[ov]
        co_u=cur[iu].astype(float)-cur[ou].astype(float); co_v=cur[iv].astype(float)-cur[ov].astype(float)
        ro=(1-t)*ro_u+t*ro_v; co=(1-t)*co_u+t*co_v
        out[layer]={
          'endpoint_rows':[iu,iv],
          'offset_from_O_reference_endpoints_m':[ro_u.tolist(),ro_v.tolist()],
          'offset_from_O_current037_endpoints_m':[co_u.tolist(),co_v.tolist()],
          'interpolated_offset_reference_m':ro.tolist(),
          'interpolated_offset_current037_m':co.tolist(),
          'proposed_reference_position_m':(np.asarray(midpoint_reference_m) if layer=='O' else np.asarray(midpoint_reference_m)+ro).tolist(),
          'proposed_current037_position_m_f64':(np.asarray(midpoint_current037_m) if layer=='O' else np.asarray(midpoint_current037_m)+co).tolist()
        }
    return out

def parse_detail(path):
 raw=path.read_bytes();magic,abi,nb,nv,ni,_=struct.unpack_from('<8s5I',raw)
 if magic!=b'NHSKIN1\0' or abi!=5 or ni%3:raise RuntimeError('NHSKIN ABI mismatch')
 vo=60+nb*36;io=vo+nv*56;wo=io+ni*4
 if len(raw)!=wo+nv*nb*4:raise RuntimeError('NHSKIN payload length mismatch')
 faces=np.frombuffer(raw,dtype='<u4',count=ni,offset=io).reshape(-1,3).astype(np.int64,copy=True)
 return faces,{'abi':int(abi),'binding_count':int(nb),'vertex_count':int(nv),'face_count':int(ni//3)}

def run():
 t0=time.monotonic()
 pins={'compact':sha(COMPACT),'edge_report':sha(EDGE_REPORT),'transfer':sha(TRANSFER/'Wnew-detail-to-084-nodes.npz'),'detail':sha(DETAIL),'rows':sha(ROWS)}
 for k,v in pins.items():
  if EXPECTED[k]!=v:raise RuntimeError(f'{k} source pin mismatch {v}!={EXPECTED[k]}')
 with np.load(COMPACT,allow_pickle=False) as z:
  ref=np.asarray(z['reference_points_m'],dtype=np.float64);cur=np.asarray(z['current037_points_m_f32'])
  srcrows=np.asarray(z['source_input_row'],dtype=np.int64);global_ids=np.asarray(z['source_global_node_ids'],dtype=np.int64)
  source_o_faces=np.asarray(z['source_O_faces'],dtype=np.int64)
 with np.load(TRANSFER/'Wnew-detail-to-084-nodes.npz',allow_pickle=False) as z:
  W=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(int(x) for x in z['shape']))
 detail_faces,detail_header=parse_detail(DETAIL)
 rows=list(csv.DictReader(ROWS.open(newline='')))
 if len(rows)!=W.shape[0] or W.shape[0]!=detail_header['vertex_count']:raise RuntimeError('detail/W/row domain mismatch')
 detail_xyz=np.asarray([[float(r[f'detail_rest_world_{a}_m']) for a in 'xyz'] for r in rows],dtype=np.float64)
 active=np.asarray([int(r['active']) for r in rows],bool)
 selected_face=np.asarray([int(r['selected_face_row']) for r in rows],np.int64)
 donor=np.asarray([int(r['detail_donor_body']) for r in rows],np.int64)
 transfer_donor=np.asarray([int(r['transfer_donor_body']) for r in rows],np.int64)
 support=[];support_weight=[]
 for i in range(W.shape[0]):
  lo,hi=W.indptr[i:i+2]; pairs=[(int(j),float(x)) for j,x in zip(W.indices[lo:hi],W.data[lo:hi]) if abs(float(x))>SUPPORT_TOL]
  support.append(set(j for j,_ in pairs));support_weight.append(dict(pairs))
 # Exact direct anchor map: unique one-hot detail row per coarse source row.
 onehot=defaultdict(list)
 for i,d in enumerate(support):
  if len(d)==1:
   col=next(iter(d));w=support_weight[i][col]
   if abs(w-1.0)<=1e-7:onehot[col].append(i)
 # Source O face labels from accepted coarse face source rows.
 coarse_faces=[]
 with FACES_CSV.open(newline='') as f:
  for r in csv.DictReader(f):coarse_faces.append([int(r[f'vertex{i}']) for i in range(3)])
 if len(coarse_faces)!=len(source_o_faces):raise RuntimeError('source O face count mismatch')
 source_edges_to_faces=defaultdict(list)
 for fr,tri in enumerate(source_o_faces):
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):source_edges_to_faces[tuple(sorted((int(a),int(b))))].append(int(fr))
 marker_map={}
 with MARKER_MAP.open(newline='') as f:
  for r in csv.DictReader(f):marker_map[int(r['marker'])]=r
 # Use only edges with exact baseline O-O hits, then exclude the two non-surface volume chords.
 with EDGE_REPORT.open() as f: edge_report=json.load(f)
 edge_banks=defaultdict(set)
 for w in edge_report['witnesses']:
  u,v=map(int,w['endpoint_query_rows'])
  if u>=3*COLUMN_COUNT or v>=3*COLUMN_COUNT:continue
  if int(srcrows[u])<0 or int(srcrows[v])<0 or int(srcrows[u])>=COLUMN_COUNT or int(srcrows[v])>=COLUMN_COUNT:continue
  edge=tuple(sorted((u,v)));edge_banks[edge].add((int(w['body']),int(w['bone_stable_id'])))
 outer_edge_set=set()
 for face in source_o_faces:
  for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):outer_edge_set.add(tuple(sorted((int(a),int(b)))) )
 interior=sorted(e for e in edge_banks if e not in outer_edge_set)
 edges=sorted(e for e in edge_banks if e in outer_edge_set)
 if len(edge_banks)!=147 or len(interior)!=2 or len(edges)!=145:raise RuntimeError(f'expected 147 O-O edges ->145 boundary+2 chord; got groups={len(edge_banks)} outer={len(edges)} interior={len(interior)}')
 # Precompute NHSKIN edge incidence and support-classified detailed edge groups.
 detail_edge_faces=defaultdict(list)
 for fi,tri in enumerate(detail_faces):
  a,b,c=map(int,tri)
  for x,y in ((a,b),(b,c),(c,a)):detail_edge_faces[tuple(sorted((x,y)))].append(int(fi))
 support_to_edges=defaultdict(list)
 for a,b in detail_edge_faces:
  su=support[a]|support[b]
  if 0<len(su)<=3:support_to_edges[tuple(sorted(su))].append((a,b))
 # Face rows' vertex and weight support maps.
 face_allowed_vertices=defaultdict(set)
 for i,r in enumerate(rows):
  if active[i] and 0<=selected_face[i]<len(source_o_faces):face_allowed_vertices[int(selected_face[i])].add(i)
 # Direct anchors included in any adjacent parent face even if its selected-face field chose a neighbor.
 anchor_options={k:sorted(set(v)) for k,v in onehot.items()}
 # Oriented normals of registered source O faces and detail faces.
 coarse_face_normals=[]
 for tri in ref[source_o_faces]:
  n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);nn=np.linalg.norm(n);coarse_face_normals.append(n/nn if nn>0 else np.zeros(3))
 coarse_face_normals=np.asarray(coarse_face_normals)
 detail_face_normals=[];detail_face_support=[]
 for tri in detail_faces:
  vv=np.asarray(tri,dtype=np.int64);n=np.cross(detail_xyz[vv[1]]-detail_xyz[vv[0]],detail_xyz[vv[2]]-detail_xyz[vv[0]]);nn=np.linalg.norm(n)
  detail_face_normals.append(n/nn if nn>0 else np.zeros(3));detail_face_support.append(support[int(vv[0])]|support[int(vv[1])]|support[int(vv[2])])
 detail_face_normals=np.asarray(detail_face_normals)
 # Area-weighted vertex normals, normalized, for side/orientation check.
 vnorm=np.zeros_like(detail_xyz)
 for fi,tri in enumerate(detail_faces):
  tri=np.asarray(tri,dtype=np.int64);n=np.cross(detail_xyz[tri[1]]-detail_xyz[tri[0]],detail_xyz[tri[2]]-detail_xyz[tri[0]])
  for vi in tri:vnorm[vi]+=n
 vn=np.linalg.norm(vnorm,axis=1);vnorm[vn>0]/=vn[vn>0,None]
 output=[];status_counts=Counter()

 for edge in edges:
  u,v=edge;parent_faces=source_edges_to_faces.get(edge,[])
  rec={'edge_output_rows':[u,v],'edge_global_node_ids':[int(global_ids[u]),int(global_ids[v])],
       'crossing_banks':[{'body_index':a,'bone_stable_id':b} for a,b in sorted(edge_banks[edge])],
       'parent_source_O_face_rows':parent_faces,'status':None,'proposal_eligible':False}
  if len(parent_faces)!=2:
   rec.update(status='fail_nonmanifold_or_missing_source_parent_faces',reason=f'expected two source O parents, got {len(parent_faces)}');output.append(rec);status_counts[rec['status']]+=1;continue
  aopts=list(anchor_options.get(u,[]));bopts=list(anchor_options.get(v,[]))
  anchor_method='unique_W_onehot'
  if not aopts or not bopts:
   # Source-face constrained inverse anchor: pick an active detailed row whose exact W support
   # contains the coarse endpoint, then minimize registered-rest distance; retain ambiguity.
   for coarse_row,opts,other in ((u,aopts,parent_faces),(v,bopts,parent_faces)):
    if opts: continue
    candidates=[]
    for dr in range(len(rows)):
     if not active[dr] or not (support[dr] & {coarse_row}): continue
     if int(selected_face[dr]) not in other: continue
     candidates.append(dr)
    if candidates:
     candidates.sort(key=lambda dr:(float(np.linalg.norm(detail_xyz[dr]-ref[coarse_row])),int(rows[dr]['detail_source_vertex_id']),dr))
     if coarse_row==u:aopts=candidates[:1]
     else:bopts=candidates[:1]
     anchor_method='source_face_supported_detail_feature'
  if not aopts or not bopts:
   rec.update(status='fail_no_source_supported_detail_anchor',reason='no direct or exact source-face-supported detail anchor',anchor_options=[aopts,bopts],
     onehot_counts_by_endpoint=[len(onehot.get(u,[])),len(onehot.get(v,[]))]);output.append(rec);status_counts[rec['status']]+=1;continue
  all_anchor_rows=sorted(set(aopts+bopts))
  endpoint_donors=set()
  for dr in all_anchor_rows:
   endpoint_donors.add(int(donor[dr]));endpoint_donors.add(int(transfer_donor[dr]))
  # Add donor IDs represented on the two exact registered parent faces, allowing only local
  # source-supported handoff across an adjacent donor seam.
  parent_rows=set()
  for fr in parent_faces: parent_rows.update(face_allowed_vertices.get(fr,set()))
  allowed_donors=set(endpoint_donors)
  for dr in parent_rows:
   allowed_donors.add(int(donor[dr]));allowed_donors.add(int(transfer_donor[dr]))
  # Actual registered detailed NHSKIN edge graph. Restrict to active rows and the local
  # endpoint/parent-face donor set; do not require every path edge's W support to stay in
  # the two coarse triangles, since that excludes valid curved registered source paths.
  graph=defaultdict(list)
  for (aa,bb),fis in detail_edge_faces.items():
   if not active[aa] or not active[bb]: continue
   if int(donor[aa]) not in allowed_donors or int(donor[bb]) not in allowed_donors: continue
   if int(transfer_donor[aa]) not in allowed_donors or int(transfer_donor[bb]) not in allowed_donors: continue
   length=float(np.linalg.norm(detail_xyz[aa]-detail_xyz[bb]))
   if length<=0:continue
   graph[aa].append((bb,length));graph[bb].append((aa,length))
  choices=[]
  for start in aopts:
   for goal in bopts:
    if start==goal: continue
    pth,spath=shortest_path_counted(graph,start,goal)
    if pth is not None:
     choices.append((float(spath['distance_m']),tuple(int(rows[x]['detail_source_vertex_id']) for x in pth),int(start),int(goal),pth,spath))
  if not choices:
   rec.update(status='fail_donor_restricted_surface_graph_disconnected',reason='no path in active registered NHSKIN graph restricted to endpoint and exact-parent-face donor IDs',
      endpoint_detail_row_options=aopts+bopts,endpoint_donor_ids=sorted(endpoint_donors),allowed_local_donor_ids=sorted(allowed_donors),
      candidate_graph_vertices=len(graph),candidate_graph_edges=sum(len(x) for x in graph.values())//2);output.append(rec);status_counts[rec['status']]+=1;continue
  choices.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
  best=choices[0];start,goal,pth,spath=best[2],best[3],best[4],best[5]
  xyz=detail_xyz[np.asarray(pth,dtype=np.int64)];mid=path_midpoint(xyz);path_rows=[int(x) for x in pth]
  path_edge_records=[];face_cos=[];edge_parities=[];orientation_unresolved=[]
  for aa,bb in zip(path_rows[:-1],path_rows[1:]):
   e=tuple(sorted((aa,bb)));face_ids=detail_edge_faces.get(e,[]);face_checks=[]
   for fi in face_ids:
    tri=list(map(int,detail_faces[fi]))
    selected=[int(selected_face[q]) for q in tri if 0<=int(selected_face[q])<len(source_o_faces)]
    # Prefer a common mapped source face; otherwise retain the full selected-face lineage.
    common=set(selected) if selected else set()
    for q in tri:
     sf=int(selected_face[q])
     if 0<=sf<len(source_o_faces):common &= {sf}
     else:common=set()
    source_candidates=sorted(common) if common else sorted(set(selected))
    checks=[]
    fn=detail_face_normals[fi]
    for sf in source_candidates:
     cn=coarse_face_normals[sf]
     cos=float(np.dot(fn,cn))
     if math.isfinite(cos) and abs(cos)>1e-8:
      checks.append({'source_face_row':sf,'signed_cosine':cos,'orientation_parity':1 if cos>0 else -1})
    if checks:
     checks.sort(key=lambda q:(-abs(q['signed_cosine']),q['source_face_row']))
     chosen=checks[0];face_cos.append(chosen['signed_cosine']);face_checks.append({'detail_face_row':int(fi),'source_face_checks':checks,'chosen_source_face_row':chosen['source_face_row'],'chosen_signed_cosine':chosen['signed_cosine'],'chosen_orientation_parity':chosen['orientation_parity']})
   if not face_checks:
    orientation_unresolved.append({'detail_edge_rows':[aa,bb],'incident_detail_faces':face_ids,'reason':'no reliable oriented source-face comparison'})
   signs={int(x['chosen_orientation_parity']) for x in face_checks}
   if len(signs)==1: edge_parities.append(next(iter(signs)))
   elif len(signs)>1: edge_parities.append(0)
   else: edge_parities.append(None)
   path_edge_records.append({'detail_edge_rows':[aa,bb],'detail_face_orientation_checks':face_checks})
  nonzero=[x for x in edge_parities if x in (-1,1)]
  parity_consistent=(len(nonzero)==len(edge_parities) and len(set(nonzero))==1 and not orientation_unresolved)
  endpoint_normal_cos=[]
  for coarse_row,anchor in zip((u,v),(start,goal)):
   inds=np.flatnonzero(np.any(source_o_faces==coarse_row,axis=1));nn=np.zeros(3)
   for fr in inds:
    tri=ref[source_o_faces[fr]];nn+=np.cross(tri[1]-tri[0],tri[2]-tri[0])
   cn=np.linalg.norm(nn);dn=np.linalg.norm(vnorm[anchor]);cos=float(np.dot(nn/cn,vnorm[anchor])) if cn>0 and dn>0 else float('nan')
   endpoint_normal_cos.append({'coarse_output_row':coarse_row,'detail_anchor_row':anchor,'signed_cosine_detail_to_coarse_vertex_normal':cos,'orientation_parity':(1 if cos>0 else -1 if cos<0 else None)})
  endpoint_signs={x['orientation_parity'] for x in endpoint_normal_cos}
  endpoint_parity_consistent=(None not in endpoint_signs and len(endpoint_signs)==1)
  normal_compatible=parity_consistent and endpoint_parity_consistent and (not nonzero or next(iter(set(nonzero)))==next(iter(endpoint_signs)))
  # Half-arclength detail-edge midpoint and exact barycentric source-anchor blend.
  aa,bb=path_rows[mid['edge_index']:mid['edge_index']+2];alpha=float(mid['edge_fraction']);mw=defaultdict(float);source_bary=defaultdict(float)
  for dr,fac in ((aa,1-alpha),(bb,alpha)):
   for col,w in support_weight[dr].items():mw[int(col)]+=fac*float(w)
   rr=rows[dr]
   for k in range(3):
    cv=int(rr[f'coarse_vertex{k}']);wt=float(rr[f'weight{k}'])
    if wt:source_bary[cv]+=fac*wt
  midpoint_ref=np.asarray(mid['point_m'],dtype=np.float64)
  delta=sum((float(w)*(cur[int(col)].astype(float)-ref[int(col)]) for col,w in mw.items()),start=np.zeros(3))
  midpoint_cur=midpoint_ref+delta
  path_fraction=float(mid['fraction'])
  field=construct_midpoint_layers(midpoint_ref,midpoint_cur,ref,cur,(u,v),path_fraction)
  tie_count=max(int(x[5]['shortest_path_count_capped_2']) for x in choices if abs(x[0]-best[0])<=SHORTEST_TIE_TOL_M)
  source_face_rows=sorted({int(rows[x]['selected_face_row']) for x in (aa,bb) if 0<=int(rows[x]['selected_face_row'])<len(source_o_faces)})
  ambiguities=[]
  if tie_count>1 or len(choices)>1:ambiguities.append('shortest_path_tie_stable_detail_source_id_tiebreak')
  if len(aopts)>1 or len(bopts)>1:ambiguities.append('duplicate_direct_W_onehot_anchor_stable_id_tiebreak')
  if not normal_compatible:ambiguities.append('source_orientation_parity_unresolved_or_contradictory')
  if edge_parities and edge_parities[0]==-1:ambiguities.append('consistent_reversed_fine_vs_coarse_winding')
  status='candidate_source_path' if normal_compatible else 'candidate_source_path_orientation_unresolved'
  rec.update(status=status,proposal_eligible=bool(normal_compatible),source_kind='registered_NHSKIN_geodesic_midpoint',
    endpoint_detail_rows=[start,goal],endpoint_detail_source_vertex_ids=[int(rows[start]['detail_source_vertex_id']),int(rows[goal]['detail_source_vertex_id'])],
    endpoint_anchor_options=[[int(rows[x]['detail_source_vertex_id']) for x in aopts],[int(rows[x]['detail_source_vertex_id']) for x in bopts]],
    anchor_method=anchor_method,endpoint_detail_donor_ids=[int(donor[start]),int(donor[goal])],
    endpoint_transfer_donor_ids=[int(transfer_donor[start]),int(transfer_donor[goal])],
    allowed_local_donor_ids=sorted(allowed_donors),path_detail_rows=path_rows,
    source_path_vertex_ids=[int(rows[x]['detail_source_vertex_id']) for x in path_rows],
    path_selected_source_face_rows=sorted({int(rows[x]['selected_face_row']) for x in path_rows if 0<=int(rows[x]['selected_face_row'])<len(source_o_faces)}),
    source_face_rows=source_face_rows,edge_endpoint_source_vertex_ids=[int(global_ids[u]),int(global_ids[v])],
    path_arclength_fraction=path_fraction,path_length_m=float(mid['total_length_m']),
    midpoint_detail_edge_index=int(mid['edge_index']),midpoint_edge_fraction=alpha,
    sample_point_world_m=midpoint_ref.tolist(),midpoint_current037_world_m=midpoint_cur.tolist(),
    source_face_barycentric_support=[{'coarse_output_row':int(k),'weight':float(w)} for k,w in sorted(source_bary.items())],
    midpoint_W_displacement_support_rows=sorted(mw),
    midpoint_W_displacement_support_weights=[float(mw[x]) for x in sorted(mw)],
    midpoint_layer_positions=field,source_I_D_field_interpolation_fraction=path_fraction,
    shortest_path={'connected':True,'distance_m':best[0],'shortest_path_count_capped_2':tie_count,'tie_tolerance_m':SHORTEST_TIE_TOL_M,
      'deterministic_tiebreak':'lexicographic stable detail_source_vertex_id sequence, then endpoint IDs'},
    ambiguity_count=int(len(ambiguities)),ambiguity_flags=ambiguities,
    minimum_normal_cosine=(min(face_cos,key=lambda x:abs(x)) if face_cos else None),
    normal_orientation={'path_edge_parities':edge_parities,'consistent_path_parity':(nonzero[0] if parity_consistent else None),
      'endpoint_normal_cosines':endpoint_normal_cos,'endpoint_parity_consistent':endpoint_parity_consistent,
      'normal_compatible':normal_compatible,'unresolved_edges':orientation_unresolved,'all_face_cosines':face_cos},
    path_edge_orientation_checks=path_edge_records,
    path_vertices=[{'detail_row':r,'detail_source_vertex_id':int(rows[r]['detail_source_vertex_id']),
      'selected_parent_source_face_row':int(rows[r]['selected_face_row']),
      'source_face_barycentric':{'coarse_vertex_rows':[int(rows[r][f'coarse_vertex{k}']) for k in range(3)],
        'weights':[float(rows[r][f'weight{k}']) for k in range(3)]},
      'detail_donor_body':int(donor[r]),'transfer_donor_body':int(transfer_donor[r]),
      'world_rest_m':detail_xyz[r].tolist()} for r in path_rows])
  output.append(rec);status_counts[status]+=1
 # Keep the two interior O-O crossing chords as excluded lineage, not midpoint candidates.
 excluded=[{'edge_output_rows':list(e),'crossing_banks':[{'body_index':b,'bone_stable_id':s} for b,s in sorted(edge_banks[e])],
   'status':'excluded_internal_O_O_volume_chord'} for e in interior]
 report={'status':'CURVED_SOURCE_MIDPOINT_BATCH_DIAGNOSTIC',
  'scope':'One deterministic source-supported path/midpoint proposal per baseline exact-crossing O-O outer-shell edge; exact NHSKIN source-face graph and typed donor restrictions. No geometry is changed and no surface/volume or physical gate is claimed.',
  'input_pins':{**pins,'accepted_coarse_faces_sha256':sha(FACES_CSV),'facet_marker_map_sha256':sha(MARKER_MAP)},
  'method':{'edge_population':'unique baseline-103 O-O crossing edges on the actual source O boundary; two internal O-O volume chords are excluded','path_graph':'active registered NHSKIN payload edges, restricted to donor IDs observed at endpoints and on the two exact source O parent faces; each detail row retains its exact selected source face and barycentric provenance','donor_filter':'both detail_donor_body and transfer_donor_body at every path vertex must belong to the endpoint donor set or the local exact-parent-face donor set','path_choice':'shortest edge-length path; deterministic heap ordering and predecessor tie-break by NHSKIN row ID; up to two shortest paths within 1e-12m are reported as ambiguity, not silently hidden','normal_gate':'record signed fine-face/source-face orientation cosine and endpoint normal parity; a consistent globally reversed winding is retained as a parity, while unresolved or contradictory parity is flagged for downstream review','midpoint':'half arclength along selected detailed path; I/D points are reconstructed as the new O midpoint plus linearly interpolated existing I-O and D-O endpoint offsets at the same normalized path fraction; no thickness/depth is fabricated','W_semantics':'W is a displacement prolongation operator only; midpoint weights are recorded as displacement support, not absolute-position barycentrics'},
  'counts':{'O_O_unique_edges_with_baseline_crossings':len(edge_banks),'outer_surface_edges':len(edges),'excluded_internal_volume_chords':len(interior),'records':len(output),'statuses':dict(status_counts),'proposal_eligible':sum(bool(r.get('proposal_eligible')) for r in output),'shortest_path_ties':sum(any('shortest_path_tie' in x for x in r.get('ambiguity_flags',[])) for r in output), 'orientation_unresolved':sum(r.get('status')=='candidate_source_path_orientation_unresolved' for r in output)},
  'excluded_internal_chords':excluded,'edges':output,
  'limitations':['This is source provenance and midpoint construction data, not a certified remesh.','Path uniqueness is not required; near-equal shortest paths are recorded and resolved deterministically by stable NHSKIN row ID.','Normal compatibility is a local orientation consistency check, not anatomical truth or full face correspondence.','I/D values are endpoint-field interpolants only; all resulting cells, materials, mass, bone clearance, mapped detail, and self-intersection require independent gates.'],
  'elapsed_seconds':time.monotonic()-t0}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'report':str(OUT),'counts':report['counts'],'elapsed_seconds':report['elapsed_seconds']},indent=2))
 return report

if __name__=='__main__':run()
