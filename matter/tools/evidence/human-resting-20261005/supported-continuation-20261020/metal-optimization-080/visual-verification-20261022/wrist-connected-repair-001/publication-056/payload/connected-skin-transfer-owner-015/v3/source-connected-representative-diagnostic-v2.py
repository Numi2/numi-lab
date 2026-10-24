import json,hashlib,pathlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components,dijkstra
R=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=R/'E/connected-skin-transfer-owner-015'
p=np.load(D/'actual-013-to-294-two-way-embedding-v2.npz',allow_pickle=False)
x=p['registered_current_world_m'].astype(float); f=p['registered_faces'].astype(int); w=p['registered_weights86']; bind=p['binding_body_indices'].astype(int); donor=bind[np.argmax(w,axis=1)]
active=p['detail_to_o_active'].astype(bool); mapped=p['detail_to_o_mapped'].astype(bool); miss=np.flatnonzero(active&~mapped)
o=p['physical_outer_world_m'].astype(float); go=p['physical_outer_global_volume_node_ids']; sfrow=p['o_to_detail_source_face_rows'].astype(int); bary=p['o_to_detail_source_barycentric']; odonor=p['o_to_detail_dominant_body_indices'].astype(int)
edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]); edges.sort(axis=1); edges=np.unique(edges,axis=0); lens=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)
n=len(x); rr=np.r_[edges[:,0],edges[:,1]]; cc=np.r_[edges[:,1],edges[:,0]]; dd=np.r_[lens,lens]; g=coo_matrix((dd,(rr,cc)),shape=(n,n)).tocsr(); g.sum_duplicates(); nc,lab=connected_components(g,directed=False)
dist,pred=dijkstra(g,indices=miss,directed=False,return_predecessors=True)
af=f[sfrow]; cdist=np.linalg.norm(o[:,None,:]-x[af],axis=2); costs=dist[:,af]+cdist[None,:,:]; corner=np.argmin(costs,axis=2); perO=costs.min(axis=2)
poses=json.loads((R/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json').read_text()); bodies={int(q['body_index']):q for q in poses['all_bodies']}; xc=float(np.median(x[:,0]))
rows=[]
for k,src in enumerate(miss):
 db=int(donor[src]); center=np.array(bodies[db]['current']['position_world_m']); candidateOrder=np.argsort(perO[k],kind='stable'); selected=[]
 for oi in candidateOrder:
  cost=float(perO[k,oi])
  # graph path distance is retained as a locality diagnostic; owner enforces projection cap.
  j=int(corner[k,oi]); anch=int(af[oi,j]); back=[int(anch)]; cur=int(anch)
  while cur!=int(src):
   cur=int(pred[k,cur])
   if cur<0 or len(back)>n: raise RuntimeError('broken predecessor')
   back.append(cur)
  path=list(reversed(back))
  pp=x[path]
  # body 102 is clearly on +x side; donor10 center is effectively midline and side-neutral.
  if db==102 and (np.min(pp[:,0])<=xc or o[oi,0]<=xc): continue
  selected.append((oi,j,anch,path,cost))
 if not selected: rows.append({'source_vertex':int(src),'donor':db,'eligible':False,'min_unrestricted_m':float(perO[k].min())}); continue
 oi,j,anch,path,cost=selected[0]; direct=float(np.linalg.norm(x[src]-o[oi])); euclid=float(np.linalg.norm(x[src]-x[anch])+np.linalg.norm(x[anch]-o[oi]))
 rows.append({'source_vertex':int(src),'donor':db,'eligible':True,'representative_o_local':int(oi),'representative_o_global':int(go[oi]),'representative_o_donor':int(odonor[oi]),'anchor_source_face_row':int(sfrow[oi]),'anchor_source_face_vertices':af[oi].tolist(),'anchor_vertex':anch,'anchor_barycentric':bary[oi].tolist(),'source_graph_path_vertex_ids':path,'graph_plus_anchor_connector_m':cost,'direct_source_to_o_vertex_m':direct,'path_euclidean_upper_m':euclid,'path_vertex_count':len(path),'source_component':int(lab[src]),'anchor_component':int(lab[anch]),'path_x_min_m':float(x[path,0].min()),'path_x_max_m':float(x[path,0].max()),'representative_x_side':int(np.sign(o[oi,0]-xc))})
report={'schema':'numi.human.connected-skin-transfer-015.source-connected-representatives.v2','source_payload_sha256':'d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04','candidate013_sha256':'1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1','embedding_v2_sha256':hashlib.sha256((D/'actual-013-to-294-two-way-embedding-v2.npz').read_bytes()).hexdigest(),'registered_position_sha256':'ecb83d7fb712e6d1f8b878e60cce25f28cfe8b624f7deab6cc8f53875ce707ea','graph':{'vertices':n,'edges':len(edges),'components':int(nc),'unsupported_component_counts':{str(int(q)):int(np.sum(lab[miss]==q)) for q in np.unique(lab[miss])},'source_x_median_m':xc,'edge_metric':'Euclidean lengths over exact source294 registered faces'},'selection':{'max_owner_one_ring_projection_m':0.05,'graph_path_cap':'none; graph path and connector distance are reported as locality diagnostics; existing owner maxProjection=0.05 m is enforced on one-ring projection','region':'same registered-face connected component; donor102 requires +x path and representative, donor10 is side-neutral because its current body center is within 10mm of source x median','deterministic_tie':'stable lowest physical O local row after graph-plus-anchor cost ordering; face corner order on equal corner costs','owner':'AcceptedFEMSkinTransfer::detailRepresentativeCoarseIndex, sourceConnectedFaceRings=1; one-ring projection <= representative Euclidean parent distance; donor identity remains unchanged'},'rows':rows}
out=D/'representative-source-graph-diagnostic-v2.json'; out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'report':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'graph':report['graph'],'eligible':sum(r['eligible'] for r in rows),'rows':rows},indent=2))
