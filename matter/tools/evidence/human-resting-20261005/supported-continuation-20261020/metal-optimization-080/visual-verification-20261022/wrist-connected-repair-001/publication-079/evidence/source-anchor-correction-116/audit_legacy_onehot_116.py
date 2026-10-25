#!/Users/n/numi-human-prep-venv-20261005/bin/python
import csv,gzip,json,hashlib,importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
D=Path('/Users/n/Work/numi-human-connected-surface-069/digit-source-diagnosis-110/source-anchor-correction-116')
WORK=Path('/Users/n/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
COARSE=WORK/'publication-071/evidence/exact-transfer-075/run-001/source-preflight/accepted-coarse-reference.csv.gz'
ROWS=E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv'
OLD=WORK/'digit-source-diagnosis-110/batch-146-curved-source-midpoints-quality-114.json'
WFILE=WORK/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz'
OUT=D/'legacy-W-onehot-anchor-audit.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with gzip.open(COARSE,'rt',newline='') as f:coarse=list(csv.DictReader(f))
rows=list(csv.DictReader(ROWS.open(newline='')))
xyz=np.asarray([[float(r[f'detail_rest_world_{a}_m']) for a in 'xyz'] for r in rows])
idmap={int(r['detail_source_vertex_id']):i for i,r in enumerate(rows)}
with np.load(WFILE,allow_pickle=False) as z:W=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(int(x) for x in z['shape']))
onehot=defaultdict(list)
for i in range(W.shape[0]):
 lo,hi=W.indptr[i:i+2]
 nz=[(int(j),float(w)) for j,w in zip(W.indices[lo:hi],W.data[lo:hi]) if abs(float(w))>1e-10]
 if len(nz)==1 and abs(nz[0][1]-1)<=1e-7:onehot[nz[0][0]].append(i)
old=json.loads(OLD.read_text()); recs=[];dists=[];closest=[]
for ei,e in enumerate(old['edges']):
 path=e.get('path_detail_rows') or []
 for ep,cr in enumerate(map(int,e['edge_output_rows'])):
  r=coarse[cr];sid=int(r['source_direct_id']);cw=np.asarray([float(r[f'world_{a}_m']) for a in 'xyz'])
  if sid!=4294967295:
   ar=idmap[sid];ap=xyz[ar];method='source_direct_id'
  else:
   vids=[int(r[f'source_face_v{k}']) for k in range(3)];bw=np.asarray([float(r[f'source_bary{k}']) for k in range(3)])
   arrows=[idmap[x] for x in vids];ap=bw@xyz[arrows];method='source_face_barycentric'
  candidates=onehot.get(cr,[])
  ranked=sorted(((float(np.linalg.norm(xyz[k]-cw)),int(k),float(np.linalg.norm(xyz[k]-ap))) for k in candidates))
  oldrow=int(path[0] if ep==0 else path[-1]) if path else None
  olddist=float(np.linalg.norm(xyz[oldrow]-ap)) if oldrow is not None else None
  best=ranked[0] if ranked else None
  rec={'edge_index':ei,'edge_output_rows':e['edge_output_rows'],'endpoint_index':ep,'coarse_row':cr,'source_direct_id':None if sid==4294967295 else sid,
    'authoritative_anchor_kind':method,'authoritative_source_anchor_detail_rows':([ar] if method=='source_direct_id' else arrows),
    'authoritative_source_anchor_m':ap.tolist(),'coarse_world_m':cw.tolist(),'source_anchor_to_coarse_world_residual_m':float(np.linalg.norm(ap-cw)),
    'W_onehot_candidate_count':len(candidates),'W_onehot_nearest_to_coarse_world':({'detail_row':best[1],'detail_source_vertex_id':int(rows[best[1]]['detail_source_vertex_id']),'distance_to_coarse_world_m':best[0],'distance_to_authoritative_anchor_m':best[2]} if best else None),
    'all_W_onehot_candidates': [{'detail_row':x[1],'detail_source_vertex_id':int(rows[x[1]]['detail_source_vertex_id']),'distance_to_coarse_world_m':x[0],'distance_to_authoritative_anchor_m':x[2]} for x in ranked],
    'old_selected_path_endpoint_detail_row':oldrow,'old_path_endpoint_to_authoritative_anchor_m':olddist}
  recs.append(rec)
  if best:dists.append(best[2]);closest.append(best[0])
result={'status':'W_ONEHOT_IS_NOT_GEOMETRIC_ANCHOR_AUDIT','scope':'All 292 endpoints of frozen 146 source-path proposal edges. W onehot rows are reported only as diagnostics and never chosen as geometric anchors.',
 'input_pins':{'accepted_coarse_reference_sha256':sha(COARSE),'accepted_transfer_rows_sha256':sha(ROWS),'Wnew_sha256':sha(WFILE),'old_146_batch_sha256':sha(OLD)},
 'counts':{'endpoint_rows':len(recs),'direct_id':sum(x['source_direct_id'] is not None for x in recs),'virtual_bary':sum(x['source_direct_id'] is None for x in recs),'endpoints_with_W_onehot_candidates':sum(x['W_onehot_candidate_count']>0 for x in recs),'endpoints_best_W_onehot_more_than_1mm_from_source_anchor':sum(bool(x['W_onehot_nearest_to_coarse_world'] and x['W_onehot_nearest_to_coarse_world']['distance_to_authoritative_anchor_m']>.001) for x in recs)},
 'nearest_onehot_distance_to_anchor_m':{'max':max(dists) if dists else None,'p95':float(np.percentile(dists,95)) if dists else None,'median':float(np.median(dists)) if dists else None},
 'rows':recs}
OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(OUT),'sha256':sha(OUT),'counts':result['counts'],'nearest_onehot_distance_to_anchor_m':result['nearest_onehot_distance_to_anchor_m']},indent=2))
