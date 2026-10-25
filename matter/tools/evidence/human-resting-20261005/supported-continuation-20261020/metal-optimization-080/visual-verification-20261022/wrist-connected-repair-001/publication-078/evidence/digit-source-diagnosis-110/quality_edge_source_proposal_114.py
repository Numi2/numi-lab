#!/Users/n/numi-human-prep-venv-20261005/bin/python
"""Append one quality-driven source-supported O-edge proposal to frozen 145 batch."""
from pathlib import Path
import hashlib,importlib.util,csv,json
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix

D=Path('/Users/n/Work/numi-human-connected-surface-069/digit-source-diagnosis-110')
V2=D/'source_path_batch_110_v2.py'; ORIENT=D/'orientation_scalar_audit_112.py'
BASE=D/'source-path-orientation-audit-112.json'; BATCH=D/'batch-145-curved-source-midpoints-004.json'
COARSE=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/accepted-transfer-row-export-075/run-001/source-preflight/accepted-coarse-reference.csv')
OUT=D/'batch-146-curved-source-midpoints-quality-114.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 b=loadmod('batch110v2',V2);o=loadmod('orientation112',ORIENT)
 compact=b.COMPACT; transfer=b.TRANSFER/'Wnew-detail-to-084-nodes.npz'
 with np.load(compact,allow_pickle=False) as z:
  R=np.asarray(z['reference_points_m'],dtype=np.float64);C=np.asarray(z['current037_points_m_f32']);F=np.asarray(z['source_O_faces'],dtype=np.int64);gids=np.asarray(z['output_node_ids'],dtype=np.int64)
 with np.load(transfer,allow_pickle=False) as z: W=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(int(x) for x in z['shape']))
 faces,h=b.parse_detail(b.DETAIL);rows=list(csv.DictReader(b.ROWS.open(newline='')));coarse_ref=list(csv.DictReader(COARSE.open(newline='')))
 xyz=np.asarray([[float(r[f'detail_rest_world_{a}_m']) for a in 'xyz'] for r in rows],dtype=np.float64)
 active=np.asarray([int(r['active']) for r in rows],dtype=bool);sel=np.asarray([int(r['selected_face_row']) for r in rows],dtype=np.int64)
 donor=np.asarray([int(r['detail_donor_body']) for r in rows],dtype=np.int64);tdonor=np.asarray([int(r['transfer_donor_body']) for r in rows],dtype=np.int64)
 support=[];sw=[]
 for dr in range(W.shape[0]):
  lo,hi=W.indptr[dr:dr+2];pairs=[(int(j),float(v)) for j,v in zip(W.indices[lo:hi],W.data[lo:hi]) if abs(float(v))>b.SUPPORT_TOL]
  support.append(set(j for j,_ in pairs));sw.append(dict(pairs))
 edge=(344,2332); parent=[int(i) for i,tri in enumerate(F) if set(edge).issubset(set(map(int,tri)))]
 if len(parent)!=2: raise RuntimeError(f'quality edge must have exactly two O-shell parent faces; got {parent}')
 edge_incidence=sum(edge[0] in tri and edge[1] in tri for tri in F)
 if edge_incidence!=2: raise RuntimeError('quality edge incidence mismatch')
 # Exact endpoint source maps are taken from the accepted coarse-reference export.
 onehot=defaultdict(list)
 for dr,sup in enumerate(support):
  if len(sup)==1:
   col=next(iter(sup))
   if abs(sw[dr].get(col,0.)-1.)<=1e-7: onehot[col].append(dr)
 start_options=sorted(onehot.get(edge[0],[]))
 if not start_options: raise RuntimeError('missing direct detail anchor for row 344')
 start=start_options[0]
 endpoint_records=[]
 for coarse_row in edge:
  rr=coarse_ref[coarse_row]
  ids=[int(rr[f'source_face_v{i}']) for i in range(3)]
  bary_w=np.asarray([float(rr[f'source_bary{i}']) for i in range(3)],dtype=np.float64)
  if any(i<0 or i>=len(rows) for i in ids) or abs(float(bary_w.sum())-1.)>1e-10 or np.min(bary_w)<-1e-10:
   raise RuntimeError(f'invalid accepted coarse source-face barycentric anchor for {coarse_row}')
  target=bary_w@xyz[ids]
  source_world=np.asarray([float(rr[f'world_{a}_m']) for a in 'xyz'])
  endpoint_records.append({'coarse_row':coarse_row,'source_face_vertex_rows':ids,'barycentric':bary_w,'source_point_m':target,
    'source_world_m':source_world,'source_point_residual_m':float(np.linalg.norm(target-source_world)),'donor_body':int(rr['donor_body'])})
 target_record=endpoint_records[1];target_ids=target_record['source_face_vertex_rows'];target_bary=target_record['barycentric'];target_point=target_record['source_point_m']
 target_face=tuple(target_ids)
 target_face_rows=[int(fi) for fi,tri in enumerate(faces) if set(map(int,tri))==set(target_face)]
 if len(target_face_rows)!=1:raise RuntimeError(f'expected exact detailed source face for row2332 bary anchor, got {target_face_rows}')
 target_face_row=target_face_rows[0]
 parent_rows={dr for dr in range(len(rows)) if active[dr] and int(sel[dr]) in parent}
 allowed={int(donor[start]),int(tdonor[start]),int(target_record['donor_body'])}
 for dr in parent_rows:allowed.update((int(donor[dr]),int(tdonor[dr])))
 for dr in target_ids:allowed.update((int(donor[dr]),int(tdonor[dr])))
 df=defaultdict(list)
 for fi,tri in enumerate(faces):
  aa,bb,cc=map(int,tri)
  for x,y in ((aa,bb),(bb,cc),(cc,aa)):df[tuple(sorted((x,y)))].append(fi)
 graph=defaultdict(list)
 for (aa,bb) in df:
  if not active[aa] or not active[bb] or donor[aa] not in allowed or donor[bb] not in allowed or tdonor[aa] not in allowed or tdonor[bb] not in allowed:continue
  length=float(np.linalg.norm(xyz[aa]-xyz[bb]))
  if length>0:graph[aa].append((bb,length));graph[bb].append((aa,length))
 path_choices=[]
 for goal in target_ids:
  if not active[goal]:continue
  pth,info=b.shortest_path_counted(graph,start,int(goal))
  if pth is None:continue
  total=float(info['distance_m'])+float(np.linalg.norm(xyz[goal]-target_point))
  path_choices.append((total,tuple(int(rows[q]['detail_source_vertex_id']) for q in pth),int(goal),pth,info))
 if not path_choices:raise RuntimeError('no active source-surface path to any vertex of exact row2332 source face')
 path_choices.sort(key=lambda x:(x[0],x[1],x[2]))
 total_path_length,_,goal,path,spath=path_choices[0]
 poly=np.vstack([xyz[np.asarray(path,dtype=np.int64)],target_point])
 mid=b.path_midpoint(poly);k=int(mid['edge_index']);alpha=float(mid['edge_fraction']);P=np.asarray(mid['point_m'])
 virtual_segment=(k==len(path)-1)
 a=int(path[k])
 bb=None if virtual_segment else int(path[k+1])
 # Interpolate W displacement at path sample; a final virtual endpoint uses exact face barycentrics.
 virtual_w=defaultdict(float)
 for dr,bw in zip(target_ids,target_bary):
  for col,w in sw[dr].items():virtual_w[int(col)]+=float(bw)*float(w)
 mw=defaultdict(float)
 first_support=sw[a]
 second_support=virtual_w if virtual_segment else sw[bb]
 for col,w in first_support.items():mw[int(col)]+=(1-alpha)*float(w)
 for col,w in second_support.items():mw[int(col)]+=alpha*float(w)
 Pc=P+sum((w*(C[col].astype(np.float64)-R[col]) for col,w in mw.items()),start=np.zeros(3))
 # BFS-coherent fine normals; component outward sign is anchored by O->I semantics in 112.
 flips,components,top=o.orient_faces_by_edge_adjacency(faces)
 comp=0; orient=json.loads(BASE.read_text()); sign=int(orient['component_outward_sign_from_layer_order'][str(comp)]['chosen_outward_sign'])
 fvn_r,ffn_r,fcross_r=o.area_normals(xyz,faces,flips)
 curxyz=xyz+W.dot(C.astype(np.float64)-R)
 fvn_c,ffn_c,fcross_c=o.area_normals(curxyz,faces,flips)
 incident=[target_face_row] if virtual_segment else df[tuple(sorted((a,bb)))]
 n=np.sum(fcross_r[incident],axis=0);nn=float(np.linalg.norm(n));nc=np.sum(fcross_c[incident],axis=0);ncn=float(np.linalg.norm(nc))
 if nn<=1e-30 or ncn<=1e-30:raise RuntimeError('degenerate fine path edge normal')
 n=sign*n/nn;nc=sign*nc/ncn
 endpoint_normals_r=[sign*fvn_r[start],sign*(target_bary@fvn_r[np.asarray(target_ids,dtype=np.int64)])]
 endpoint_normals_c=[sign*fvn_c[start],sign*(target_bary@fvn_c[np.asarray(target_ids,dtype=np.int64)])]
 endpoint_normals_r[1]/=np.linalg.norm(endpoint_normals_r[1]);endpoint_normals_c[1]/=np.linalg.norm(endpoint_normals_c[1])
 u,v=edge;sr=[];fr=[];sc=[];fc=[]
 for ei,coarse in enumerate(edge):
  nr=endpoint_normals_r[ei];ncur=endpoint_normals_c[ei]
  sr.append(float(-np.dot(R[coarse+2411]-R[coarse],nr)));fr.append(float(-np.dot(R[coarse+4822]-R[coarse+2411],nr)))
  sc.append(float(-np.dot(C[coarse+2411].astype(float)-C[coarse].astype(float),ncur)));fc.append(float(-np.dot(C[coarse+4822].astype(float)-C[coarse+2411].astype(float),ncur)))
 tsr=float(np.mean(sr));tfr=float(np.mean(fr));tsc=float(np.mean(sc));tfc=float(np.mean(fc))
 if min(tsr,tfr,tsc,tfc)<=0:raise RuntimeError(f'nonpositive source scalar depth at midpoint: {(tsr,tfr,tsc,tfc)}')
 prop_r=o.normal_scalar_layer_points(P,n,tsr,tfr);prop_c=o.normal_scalar_layer_points(Pc,nc,tsc,tfc)
 bary=[]
 for fi in incident:
  tri=list(map(int,faces[fi]));weights=[0.,0.,0.]
  if virtual_segment:
   for q,wt in zip(target_ids,target_bary):weights[tri.index(int(q))]=float(alpha*wt)
   weights[tri.index(a)]+=1-alpha
  else:
   weights[tri.index(a)]=1-alpha;weights[tri.index(bb)]=alpha
  bary.append({'detail_face_row':int(fi),'detail_triangle_rows':tri,'barycentric':weights})
 source_bary=defaultdict(float)
 for dr,wgt in ((a,1-alpha),):
  rr=rows[dr]
  for j in range(3):
   cv=int(rr[f'coarse_vertex{j}']);wt=float(rr[f'weight{j}'])
   if wt:source_bary[cv]+=wgt*wt
 if virtual_segment:
  for dr,bw in zip(target_ids,target_bary):
   rr=rows[int(dr)]
   for j in range(3):
    cv=int(rr[f'coarse_vertex{j}']);wt=float(rr[f'weight{j}'])
    if wt:source_bary[cv]+=alpha*float(bw)*wt
 else:
  rr=rows[bb]
  for j in range(3):
   cv=int(rr[f'coarse_vertex{j}']);wt=float(rr[f'weight{j}'])
   if wt:source_bary[cv]+=alpha*wt
 path_vertices=[{'detail_row':int(dr),'detail_source_vertex_id':int(rows[dr]['detail_source_vertex_id']),
   'selected_parent_source_face_row':int(sel[dr]),'source_face_barycentric':{
    'coarse_vertex_rows':[int(rows[dr][f'coarse_vertex{j}']) for j in range(3)],
    'weights':[float(rows[dr][f'weight{j}']) for j in range(3)]},
   'detail_donor_body':int(donor[dr]),'transfer_donor_body':int(tdonor[dr]),'world_rest_m':xyz[dr].tolist()} for dr in path]
 path_vertices.append({'virtual_source_face_point':True,'detail_face_vertex_rows':target_ids,'face_barycentric':target_bary.tolist(),'world_rest_m':target_point.tolist()})
 rec={'edge_output_rows':list(edge),'edge_global_node_ids':[int(gids[x]) for x in edge],
  'source_kind':'quality_driven_registered_NHSKIN_geodesic_midpoint','status':'SOURCE_PATH_PROPOSAL_ONLY','proposal_eligible':True,
  'reason':'actual two-face O-shell edge supplied as low-quality refinement target; not one of the 145 crossing edges and not a bone-crossing claim',
  'quality_witness':{'source':'root-supplied frozen 116 skin-quality witness','minimum_quality':7e-6,'edge_length_m':float(np.linalg.norm(R[u]-R[v]))},
  'parent_source_O_face_rows':parent,'edge_face_incidence':edge_incidence,'endpoint_detail_rows':[start,None],
  'endpoint_detail_source_vertex_ids':[int(rows[start]['detail_source_vertex_id']),None],
  'endpoint_detail_donor_ids':[int(donor[start]),int(target_record['donor_body'])], 'endpoint_transfer_donor_ids':[int(tdonor[start]),None],
  'endpoint_source_face_anchors':[{'coarse_row':q['coarse_row'],'source_face_vertex_rows':q['source_face_vertex_rows'],'barycentric':q['barycentric'].tolist(),'source_point_m':q['source_point_m'].tolist(),'coarse_world_m':q['source_world_m'].tolist(),'residual_m':q['source_point_residual_m'],'donor_body':q['donor_body']} for q in endpoint_records],
  'chosen_indirect_endpoint_anchor_method':'exact accepted-coarse-reference source face and barycentrics; path reaches one target face vertex then final segment lies in that exact face',
  'allowed_local_donor_ids':sorted(allowed),'path_detail_rows':[int(x) for x in path],
  'source_path_vertex_ids':[int(rows[x]['detail_source_vertex_id']) for x in path],
  'path_selected_source_face_rows':sorted({int(sel[x]) for x in path if 0<=int(sel[x])}),
  'source_face_rows':sorted({int(sel[x]) for x in (a,bb) if 0<=int(sel[x])}),
  'source_face_barycentric_support':[{'coarse_output_row':int(k0),'weight':float(w0)} for k0,w0 in sorted(source_bary.items())],
  'path_vertices':path_vertices,
  'path_length_m':float(mid['total_length_m']),'path_arclength_fraction':0.5,'midpoint_detail_edge_index':k,'midpoint_detail_edge_rows':[a,bb],'midpoint_on_virtual_face_segment':virtual_segment,
  'midpoint_edge_index':k,'midpoint_edge_alpha':alpha,'midpoint_edge_fraction':alpha,'registered_source_midpoint_m':P.tolist(),'sample_point_world_m':P.tolist(),
  'current037_source_midpoint_m':Pc.tolist(),'midpoint_current037_world_m':Pc.tolist(),
  'midpoint_detail_face_barycentric_samples':bary,'midpoint_W_displacement_support_rows':sorted(mw),'midpoint_W_displacement_support_weights':[float(mw[x]) for x in sorted(mw)],
  'component_id':comp,'detail_component':comp,'detail_adjacency_winding_conflicts':int(top['adjacency_winding_conflict_count']),'global_outward_sign_from_O_to_I_layer_order':sign,
  'fine_face_normals_at_midpoint_edge':(sign*ffn_r[incident]).tolist(),'current037_fine_face_normals_at_midpoint_edge':(sign*ffn_c[incident]).tolist(),
  'outward_fine_edge_normal_reference':n.tolist(),'outward_fine_edge_normal_current037':nc.tolist(),
  'reference_outward_normal':n.tolist(),'current037_outward_normal':nc.tolist(),'source_I_D_field_interpolation_fraction':0.5,
  'source_skin_depth_reference_endpoint_m':sr,'source_fat_depth_reference_endpoint_m':fr,'source_skin_depth_current037_endpoint_m':sc,'source_fat_depth_current037_endpoint_m':fc,
  'normal_scalar_source_proposal_reference':{**prop_r,'depths_are_interpolated_existing_scalar_fields':True},
  'normal_scalar_source_proposal_current037':{**prop_c,'depths_are_interpolated_existing_scalar_fields':True},
  'limitations':['Row 2332 has no unique one-hot detail anchor; this proposal uses the exact accepted-coarse-reference face vertices and barycentrics [3775,3776,3777], with the represented fine-surface point 1.413 mm from the compact coarse point.','Outward orientation is anchored by the global O-to-I layer semantic and detailed-face orientability; this is construction evidence only.','No exact full-bank, volume, bone-clearance, material, or self-intersection gate was run on this proposal.']}
 old=json.loads(BASE.read_text()); batch=json.loads(BATCH.read_text())
 crossing_rows=[]
 for source_row in old['edges']:
  row=dict(source_row)
  row['prior_coarse_face_orientation_status']=source_row.get('previous_compatibility_status')
  row['source_component_orientation_status']='RESOLVED_BY_NHSKIN_EDGE_ADJACENCY_AND_O_TO_I_LAYER_ANCHOR'
  row['source_normal_proposal_eligible']=bool(row.get('same_orientable_component') and row.get('reference_outward_normal') is not None and row.get('current037_outward_normal') is not None and row.get('orientation_faces_consistent'))
  row['source_normal_resolution_witness']={'component_id':row.get('detail_component'),'component_count':int(json.loads(BASE.read_text())['detail_topology']['component_count']),
    'adjacency_winding_conflicts':int(json.loads(BASE.read_text())['detail_topology']['adjacency_winding_conflict_count']),
    'global_outward_sign_from_O_to_I_layer_order':row.get('global_outward_sign_from_O_to_I_layer_order'),
    'fine_to_coarse_winding_parity':row.get('fine_to_coarse_winding_parity_from_direct_onehot_anchors')}
  crossing_rows.append(row)
 if sum(bool(x['source_normal_proposal_eligible']) for x in crossing_rows)!=145:raise RuntimeError('not all 145 source paths have a resolved component normal')
 rec['source_component_orientation_status']='RESOLVED_BY_NHSKIN_EDGE_ADJACENCY_AND_O_TO_I_LAYER_ANCHOR'
 rec['source_normal_proposal_eligible']=True
 rows_out=crossing_rows+[rec]
 if len(rows_out)!=146 or len({tuple(x['edge_output_rows']) for x in rows_out})!=146:raise RuntimeError('combined batch is not 146 unique rows')
 result={'status':'146_SOURCE_MIDPOINT_PROPOSALS_DIAGNOSTIC_ONLY','scope':'145 exact-crossing O-edge source path proposals with component-oriented normals plus one independently identified low-quality O-shell edge (344,2332). Quality edge is not counted as a bone crossing. No geometry changed and no collision/volume acceptance run.','input_pins':{'batch145_sha256':sha(BATCH),'orientation112_sha256':sha(BASE),'source_path_runner_sha256':sha(V2),'orientation_runner_sha256':sha(ORIENT),'compact_sha256':sha(compact),'transfer_sha256':sha(transfer),'detail_sha256':sha(b.DETAIL),'accepted_transfer_rows_sha256':sha(b.ROWS),'accepted_coarse_reference_sha256':sha(COARSE)},'counts':{'crossing_source_edges':145,'quality_only_source_edges':1,'total_unique_edges':146,'source_normal_orientable_paths':sum(bool(x.get('source_normal_proposal_eligible')) for x in rows_out),'145_crossing_edges_with_outward_reference_and_current_normals':145,'quality_edge_with_outward_reference_and_current_normals':1,'negative_endpoint_scalar_depth_edges_in_145':2,'quality_edge_parent_faces':2},'quality_edge_source_proposal':rec,'edges':rows_out,'limitations':['The 145 inherited rows and new quality row are source-path proposals only; exact full mesh and all material/clearance gates remain required.','No fixed layer thickness is introduced. O/I/D suggestions use existing endpoint scalar layer depths projected onto fine source normals; two crossing-edge rows retain negative endpoint depth witnesses from source data and these are not clamped.','The row2332 endpoint uses the accepted coarse-reference exact face and barycentrics; its represented detailed source point remains 1.413mm from the compact coarse point and this residual is retained.','The supplied quality threshold is retained as a root-supplied witness and is not independently recomputed in this script.']}
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'output':str(OUT),'sha256':sha(OUT),'counts':result['counts'],'quality_edge':{'path_vertices':len(path),'path_length_m':mid['total_length_m'],'midpoint':P.tolist(),'endpoint_depths_ref':[sr,fr],'source_anchor_residual_m':target_record['source_point_residual_m'],'virtual_segment':virtual_segment}},indent=2))
if __name__=='__main__':main()
