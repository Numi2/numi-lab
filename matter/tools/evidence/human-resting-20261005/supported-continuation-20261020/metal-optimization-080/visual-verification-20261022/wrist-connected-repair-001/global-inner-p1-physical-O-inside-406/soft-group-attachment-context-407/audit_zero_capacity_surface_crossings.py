#!/usr/bin/env python3
import json, glob, hashlib, os
import numpy as np
BASE='/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
CP=BASE+'/global-inner-p1-full-domain-clipping-009/run-001/checkpoint.json'
MESH=BASE+'/global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
CLASS=BASE+'/global-inner-p1-physical-O-inside-406/classification-report.json'
PARTDIR=BASE+'/global-inner-p1-full-domain-clipping-009/run-001/constraint-parts'
OUT=BASE+'/global-inner-p1-physical-O-inside-406/soft-group-attachment-context-407/full-domain-zero-capacity-crossings-009.json'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
cp=json.load(open(CP)); cls=json.load(open(CLASS)); m=np.load(MESH,allow_pickle=False)
X=m['points_world_m']; T=m['tetrahedra']; O=set(map(int,m['physical_outer_node_ids']))
outer_faces=m['physical_outer_faces']; face_set={tuple(sorted(map(int,f))) for f in outer_faces}
outer_edges=set()
for f in outer_faces:
 a,b,c=map(int,f)
 outer_edges.add(tuple(sorted((a,b)))); outer_edges.add(tuple(sorted((b,c)))); outer_edges.add(tuple(sorted((c,a))))
flagged=[]
for n in cls['nodes']:
 groups=n.get('soft_groups_containing_o',[])
 if groups:
  flagged.append({'o_node':int(n['o_node_014']),'xyz':np.asarray(n['world_m'],float),'stable_ids':[int(g['stable_id']) for g in groups]})
results=[]
for sid in (75,93,103,132):
 ps=[p for p in cp.get('parts',[]) if int(p.get('stable_id',-1))==sid and int(p.get('zero_capacity_rows',0))>0]
 rec={'stable_id':sid,'processed_parts':len(ps),'faces_processed':sum(int(p['faces_processed']) for p in ps),'zero_capacity_rows':sum(int(p['zero_capacity_rows']) for p in ps),'bank_ids':sorted(set(int(p['bank_id']) for p in ps)),'parts':[]}
 allw=[]
 for p in ps:
  q=os.path.join(PARTDIR,p['file']); z=np.load(q,allow_pickle=False)
  zr=z['zero_capacity_polygon_vertex_rows']; off=z['poly_offsets']; barys=z['polygon_barycentric'];
  polyidx=np.searchsorted(off,zr,side='right')-1
  local_face_base=int(p['flat_face_range'][0])-int(p['local_face_range'][0])
  partw=[]
  for row,pi in zip(zr,polyidx):
   tid=int(z['tet_ids'][pi]); fid=int(z['face_ids'][pi]); bc=barys[row]; nodes=T[tid]
   xyz=bc@X[nodes]
   support=tuple(int(v) for v,b in zip(nodes,bc) if b>1e-9)
   all_o=all(v in O for v in support)
   oface=(len(support)==3 and tuple(sorted(support)) in face_set)
   oedge=(len(support)==2 and tuple(sorted(support)) in outer_edges)
   near=min(((float(np.linalg.norm(xyz-v['xyz'])),v['o_node']) for v in flagged),default=(None,None))
   w={'point_world_m':xyz.tolist(),'tet_row':tid,'bank_global_face_row':fid,'bank_local_face_row':fid-local_face_base,'bank_index':int(p['bank_id']),'support_nodes':list(support),'support_all_physical_O':all_o,'exact_physical_O_face':oface,'exact_physical_O_edge':oedge,'nearest_flagged_o_node':near[1],'distance_to_nearest_flagged_o_m':near[0]}
   partw.append(w); allw.append(w)
  rec['parts'].append({'part_file':p['file'],'part_sha256':p['sha256'],'bank_index':int(p['bank_id']),'bank_flat_face_range':p['flat_face_range'],'bank_local_face_range':p['local_face_range'],'zero_capacity_rows':len(zr),'all_support_physical_O':sum(w['support_all_physical_O'] for w in partw),'exact_physical_O_faces':sum(w['exact_physical_O_face'] for w in partw),'exact_physical_O_edges':sum(w['exact_physical_O_edge'] for w in partw)})
 rec['support_all_physical_O_count']=sum(w['support_all_physical_O'] for w in allw)
 rec['exact_physical_O_face_count']=sum(w['exact_physical_O_face'] for w in allw)
 rec['exact_physical_O_edge_count']=sum(w['exact_physical_O_edge'] for w in allw)
 bynode={}
 for n in flagged:
  if sid not in n['stable_ids']: continue
  best=min(allw,key=lambda w:np.linalg.norm(np.asarray(w['point_world_m'])-n['xyz'])) if allw else None
  bynode[str(n['o_node'])]=None if best is None else {'distance_m':float(np.linalg.norm(np.asarray(best['point_world_m'])-n['xyz'])),'bank_index':best['bank_index'],'bank_local_face_row':best['bank_local_face_row'],'tet_row':best['tet_row'],'point_world_m':best['point_world_m'],'support_nodes':best['support_nodes'],'physical_O_face':best['exact_physical_O_face'],'physical_O_edge':best['exact_physical_O_edge']}
 rec['nearest_intersection_for_flagged_o_node_by_id']=bynode
 rec['nearest_12_witnesses_to_any_flagged_o']=[{'distance_m':float(min(np.linalg.norm(np.asarray(w['point_world_m'])-n['xyz']) for n in flagged)),'witness':w} for w in sorted(allw,key=lambda w:min(np.linalg.norm(np.asarray(w['point_world_m'])-n['xyz']) for n in flagged))[:12]]
 results.append(rec)
report={'schema':'numi.human.physical-O-fullmesh-zero-capacity-surface-crossings.407.v1','status':'partial-stream diagnostic; no geometry changed or admitted','interpretation':'Each zero-capacity clipped polygon vertex is reconstructed from the recorded original tet row and barycentric coordinates. A point supported only by physical-O nodes and matching an emitted physical-O face/edge is an actual bank-surface/physical-O-boundary intersection witness in this mesh. This establishes geometric intersection, not which anatomy owner is wrong; source skin undersize versus tissue shape/registration requires attachment and landmark evidence.','checkpoint':{'path':CP,'sha256':sha(CP),'complete':cp['complete'],'faces_processed':cp['faces_processed'],'next_bank':cp['next_bank'],'next_local_face':cp['next_local_face'],'zero_capacity_rows':cp['zero_capacity_rows'],'insufficient_capacity_rows':cp['insufficient_capacity_rows'],'mesh_nodes':cp['mesh_nodes'],'mesh_tets':cp['mesh_tets']},'pins':{'mesh_path':MESH,'mesh_sha256':sha(MESH),'classification_406_path':CLASS,'classification_406_sha256':sha(CLASS),'tissue_banks_path':BASE+'/registered-surrounding-tissue-native013-191/registered-tissue-banks.npz','tissue_banks_sha256':cp['inputs'].get(BASE+'/registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'),'clipper_sha256':cp['inputs'].get(BASE+'/robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py')},'target_stable_ids':[75,93,103,132],'stable_id_results':results}
os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,'w') as f: json.dump(report,f,indent=2,sort_keys=True); f.write('\n')
print(OUT); print('sha256',sha(OUT)); print('checkpoint',cp['faces_processed'],cp['next_bank'],cp['zero_capacity_rows'])
for r in results:
 print('stable',r['stable_id'],'parts',r['processed_parts'],'zero',r['zero_capacity_rows'],'allO',r['support_all_physical_O_count'],'face',r['exact_physical_O_face_count'],'edge',r['exact_physical_O_edge_count'],'nearestO',[(k,None if v is None else (round(v['distance_m']*1000,3),v['bank_local_face_row'],v['tet_row'])) for k,v in r['nearest_intersection_for_flagged_o_node_by_id'].items()])
