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
OUT=D/'batch-145-curved-source-midpoints.json'
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
 anchor_by_coarse={k:v[0] for k,v in onehot.items() if len(v)==1}
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
  anchors=[anchor_by_coarse.get(u),anchor_by_coarse.get(v)]
  if any(x is None for x in anchors):
   rec.update(status='fail_no_unique_direct_detail_anchor',reason='one or both coarse endpoints lack a unique W one-hot NHSKIN row',anchor_detail_rows=anchors,
      onehot_counts_by_endpoint=[len(onehot.get(u,[])),len(onehot.get(v,[]))]);output.append(rec);status_counts[rec['status']]+=1;continue
  start,goal=map(int,anchors)
  if start==goal:
   rec.update(status='fail_collapsed_detail_anchors',reason='coarse edge endpoints map to same detail row');output.append(rec);status_counts[rec['status']]+=1;continue
  endpoint_donors={int(donor[start]),int(donor[goal]),int(transfer_donor[start]),int(transfer_donor[goal])}
  # Build only detailed edges whose W support belongs to one of the two exact source parent faces.
  graph=defaultdict(list);edge_parent={};face_vertex_union=set()
  for fr in parent_faces:
   fset=set(map(int,source_o_faces[fr]));av=set(face_allowed_vertices.get(fr,set()))
   for cv in fset:
    ar=anchor_by_coarse.get(cv)
    if ar is not None:av.add(ar)
   face_vertex_union.update(av)
   for n in (1,2,3):
    for subset in itertools.combinations(sorted(fset),n):
     for a,b in support_to_edges.get(tuple(subset),()):
      if a not in av or b not in av:continue
      if donor[a] not in endpoint_donors or donor[b] not in endpoint_donors:continue
      if transfer_donor[a] not in endpoint_donors or transfer_donor[b] not in endpoint_donors:continue
      length=float(np.linalg.norm(detail_xyz[a]-detail_xyz[b]))
      if length<=0:continue
      key=tuple(sorted((a,b)));edge_parent.setdefault(key,set()).add(fr)
  for (a,b),frs in edge_parent.items():graph[a].append((b,float(np.linalg.norm(detail_xyz[a]-detail_xyz[b]))));graph[b].append((a,float(np.linalg.norm(detail_xyz[a]-detail_xyz[b]))))
  if start not in graph or goal not in graph:
   rec.update(status='fail_donor_restricted_surface_graph_disconnected',reason='no donor-compatible detailed path in the two exact source parent faces',
      endpoint_detail_rows=[start,goal],endpoint_donor_ids=sorted(endpoint_donors),candidate_graph_vertices=len(face_vertex_union),candidate_graph_edges=len(edge_parent));output.append(rec);status_counts[rec['status']]+=1;continue
  pth,spath=shortest_path_counted(graph,start,goal)
  if pth is None:
   rec.update(status='fail_donor_restricted_surface_graph_disconnected',reason='endpoints disconnected in donor-compatible source graph',endpoint_detail_rows=[start,goal],endpoint_donor_ids=sorted(endpoint_donors),candidate_graph_vertices=len(face_vertex_union),candidate_graph_edges=len(edge_parent));output.append(rec);status_counts[rec['status']]+=1;continue
  xyz=detail_xyz[np.asarray(pth,dtype=np.int64)];mid=path_midpoint(xyz);path_rows=[int(x) for x in pth]
  # Verify each path edge is an actual detailed NHSKIN edge with source-parent ancestry.
  path_edge_records=[];source_norm_cos=[]
  normals_ok=True;unresolved_normals=[]
  for a,b in zip(path_rows[:-1],path_rows[1:]):
   e=tuple(sorted((a,b)));face_ids=detail_edge_faces.get(e,[]);eligible=[]
   for fi in face_ids:
    for fr in parent_faces:
     fset=set(map(int,source_o_faces[fr]))
     if detail_face_support[fi] <= fset:
      cos=float(np.dot(detail_face_normals[fi],coarse_face_normals[fr]))
      source_norm_cos.append(cos);eligible.append({'detail_face_row':int(fi),'parent_source_face_row':int(fr),'normal_cosine_to_parent':cos})
   if not eligible:unresolved_normals.append({'detail_edge':[a,b],'reason':'no incident fine face fits exact parent W support'});normals_ok=False
   elif any(x['normal_cosine_to_parent']<=0 for x in eligible):normals_ok=False
   path_edge_records.append({'detail_edge_rows':[a,b],'supported_parent_face_rows':sorted(map(int,edge_parent.get(e,set()))),'incident_face_parent_normal_checks':eligible})
  endpoint_normal_cos=[]
  for coarse_row,anchor in zip((u,v),(start,goal)):
   # Compare detail anchor's oriented area-weighted normal with its coarse O one-ring normal.
   inds=np.flatnonzero(np.any(source_o_faces==coarse_row,axis=1));nn=np.zeros(3)
   for fr in inds:
    tri=ref[source_o_faces[fr]];nn+=np.cross(tri[1]-tri[0],tri[2]-tri[0])
   cn=np.linalg.norm(nn);dn=np.linalg.norm(vnorm[anchor]);cos=float(np.dot(nn/cn,vnorm[anchor])) if cn>0 and dn>0 else float('nan')
   endpoint_normal_cos.append({'coarse_output_row':coarse_row,'detail_anchor_row':anchor,'cosine_detail_anchor_normal_vs_coarse_O_vertex_normal':cos})
   if not math.isfinite(cos) or cos<=0:normals_ok=False
  # The midpoint's transfer weights are displacement support, not absolute position barycentrics.
  a,b=path_rows[mid['edge_index']:mid['edge_index']+2];alpha=float(mid['edge_fraction']);mw=defaultdict(float)
  for row,fac in ((a,1-alpha),(b,alpha)):
   for col,w in support_weight[row].items():mw[int(col)]+=fac*float(w)
  allowed_face_rows=[fr for fr in parent_faces if set(mw)<=set(map(int,source_o_faces[fr]))]
  field=edge_midpoint_field(ref,cur,u,v,.5)
  statuses=[]
  tie_count=int(spath['shortest_path_count_capped_2'])
  if tie_count>1:statuses.append('shortest_path_tie_stable_row_id_tiebreak')
  if not normals_ok:statuses.append('normal_compatibility_unresolved_or_opposed')
  status='candidate_source_path' if normals_ok else 'candidate_path_normal_gate_failed'
  rec.update(status=status,proposal_eligible=bool(normals_ok),shortest_path=spath,
    endpoint_detail_rows=[start,goal],endpoint_detail_donor_ids=[int(donor[start]),int(donor[goal])],
    allowed_donor_ids=sorted(endpoint_donors),path_detail_rows=path_rows,
    path_source_parent_face_rows=sorted({int(rows[r]['selected_face_row']) for r in path_rows}),
    path_length_m=float(mid['total_length_m']),path_arclength_fraction=float(mid['fraction']),
    midpoint_detail_edge_index=int(mid['edge_index']),midpoint_edge_fraction=float(alpha),
    midpoint_world_rest_m=mid['point_m'].tolist(),
    midpoint_W_displacement_support_rows=sorted(mw),
    midpoint_W_displacement_support_weights=[float(mw[x]) for x in sorted(mw)],
    midpoint_W_support_parent_face_rows=allowed_face_rows,
    layer_endpoint_interpolation_fraction=0.5,
    source_I_D_field_interpolation=field,
    oriented_parent_face_normal_checks={'minimum_fine_face_to_parent_cosine':(min(source_norm_cos) if source_norm_cos else None),
       'endpoint_normal_cosines':endpoint_normal_cos,'all_path_edges_have_compatible_oriented_source_faces':normals_ok,
       'unresolved_edges':unresolved_normals},
    path_edge_parentage_and_normal_checks=path_edge_records,
    ambiguity_flags=statuses,
    path_vertices=[{'detail_row':r,'selected_parent_source_face_row':int(rows[r]['selected_face_row']),
      'detail_donor_body':int(donor[r]),'transfer_donor_body':int(transfer_donor[r]),
      'world_rest_m':detail_xyz[r].tolist()} for r in path_rows])
  output.append(rec);status_counts[status]+=1
 # Keep the two interior O-O crossing chords as excluded lineage, not midpoint candidates.
 excluded=[{'edge_output_rows':list(e),'crossing_banks':[{'body_index':b,'bone_stable_id':s} for b,s in sorted(edge_banks[e])],
   'status':'excluded_internal_O_O_volume_chord'} for e in interior]
 report={'status':'CURVED_SOURCE_MIDPOINT_BATCH_DIAGNOSTIC',
  'scope':'One deterministic source-supported path/midpoint proposal per baseline exact-crossing O-O outer-shell edge; exact NHSKIN source-face graph and typed donor restrictions. No geometry is changed and no surface/volume or physical gate is claimed.',
  'input_pins':{**pins,'accepted_coarse_faces_sha256':sha(FACES_CSV),'facet_marker_map_sha256':sha(MARKER_MAP)},
  'method':{'edge_population':'unique baseline-103 O-O crossing edges on the actual source O boundary; two internal O-O volume chords are excluded','path_graph':'NHSKIN payload face edges whose W displacement supports are subsets of one of the two exact source O parent faces; active transfer rows are restricted to those selected parent faces plus unique one-hot endpoint anchors','donor_filter':'both detail_donor_body and transfer_donor_body at every path vertex must belong to the union of endpoint anchor donor IDs','path_choice':'shortest edge-length path; deterministic heap ordering and predecessor tie-break by NHSKIN row ID; up to two shortest paths within 1e-12m are reported as ambiguity, not silently hidden','normal_gate':'oriented fine face normals must have positive cosine to their supported source O parent face; detail endpoint area-weighted normals must have positive cosine to coarse O vertex one-ring normals','midpoint':'half arclength along selected detailed path; I/D field positions are interpolated from existing corresponding source endpoint rows at the same normalized path fraction 0.5; no thickness/depth is fabricated','W_semantics':'W is a displacement prolongation operator only; midpoint weights are recorded as displacement support, not absolute-position barycentrics'},
  'counts':{'O_O_unique_edges_with_baseline_crossings':len(edge_banks),'outer_surface_edges':len(edges),'excluded_internal_volume_chords':len(interior),'records':len(output),'statuses':dict(status_counts),'proposal_eligible':sum(bool(r.get('proposal_eligible')) for r in output),'shortest_path_ties':sum('shortest_path_tie_stable_row_id_tiebreak' in r.get('ambiguity_flags',[]) for r in output)},
  'excluded_internal_chords':excluded,'edges':output,
  'limitations':['This is source provenance and midpoint construction data, not a certified remesh.','Path uniqueness is not required; near-equal shortest paths are recorded and resolved deterministically by stable NHSKIN row ID.','Normal compatibility is a local orientation consistency check, not anatomical truth or full face correspondence.','I/D values are endpoint-field interpolants only; all resulting cells, materials, mass, bone clearance, mapped detail, and self-intersection require independent gates.'],
  'elapsed_seconds':time.monotonic()-t0}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'report':str(OUT),'counts':report['counts'],'elapsed_seconds':report['elapsed_seconds']},indent=2))
 return report

if __name__=='__main__':run()
