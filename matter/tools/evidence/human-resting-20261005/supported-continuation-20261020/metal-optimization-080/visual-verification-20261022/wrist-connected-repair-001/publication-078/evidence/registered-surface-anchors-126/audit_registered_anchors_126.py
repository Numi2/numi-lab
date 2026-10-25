from pathlib import Path
import csv,json,hashlib,sys,collections
import numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
P=D/'registered-surface-anchors-126/run-004';assert not P.exists();P.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(E/'E/accepted-transfer-row-adapter-075'))
from load_transfer_075 import load_transfer075
S=E/'accepted-transfer-row-export-075/run-001/source-preflight'
files={'transfer_rows':S/'accepted-transfer-rows.csv','coarse_reference':S/'accepted-coarse-reference.csv','coarse_faces':S/'accepted-coarse-faces.csv','detail_payload':E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'}
pins={'transfer_rows':'71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14','coarse_reference':'9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2','coarse_faces':'7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2','detail_payload':'7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'}
W,X,C,F=load_transfer075(files['transfer_rows'],files['coarse_reference'],files['coarse_faces'],files['detail_payload'],expected_sha256=pins)
detail_records=list(csv.DictReader(files['transfer_rows'].open()));source_ids=np.asarray([int(r['detail_source_vertex_id'])for r in detail_records],np.int64)
assert all(int(r['detail_row'])==i for i,r in enumerate(detail_records))
sys.path.insert(0,str(D/'registered-surface-anchors-126/matter/tools'))
from human_skin_fat_registered_surface_anchors import registered_surface_anchors
source_records=list(csv.DictReader(files['coarse_reference'].open()))
A,records=registered_surface_anchors(X,F,C,source_records,detail_source_vertex_ids=source_ids)
np.savez_compressed(P/'registered-anchors.npz',source_anchor_points_world_m=A,coarse_points_world_f32=C,detail_source_vertex_ids=source_ids)
onehot=collections.defaultdict(list);W=W.tocsr(copy=True);W.eliminate_zeros()
for i in range(W.shape[0]):
 a,b=W.indptr[i:i+2]
 if b-a==1 and W.data[a]==1.:onehot[int(W.indices[a])].append(i)
wrong=[]
for r in records:
 i=r['coarse_row']
 if i not in onehot:continue
 j=min(onehot[i]);err=float(np.linalg.norm(X[j].astype(float)-A[i]))
 if err>1e-6:wrong.append({'coarse_row':i,'hypothetical_min_onehot_detail_row':j,'registered_anchor_kind':r['kind'],'registered_source_ids':r['source_vertex_ids'],'distance_to_registered_anchor_m':err})
report={'input_sha256':pins,'anchor_helper_sha256':sha(D/'registered-surface-anchors-126/matter/tools/human_skin_fat_registered_surface_anchors.py'),'source_id_equals_row_verified':bool(np.array_equal(source_ids,np.arange(len(X)))),'columns':len(records),'anchor_counts':dict(collections.Counter(r['kind']for r in records)),'max_direct_residual_m':max(r['coincidence_residual_m']for r in records if r['kind']=='source_direct_id'),'max_indirect_offset_m':max(r['coincidence_residual_m']for r in records if r['kind']=='source_face_barycentric'),'minimum_onehot_is_wrong_above_1um_columns':len(wrong),'worst_hypothetical_min_onehot_error_m':max((r['distance_to_registered_anchor_m']for r in wrong),default=0),'wrong_min_onehot_examples':sorted(wrong,key=lambda r:-r['distance_to_registered_anchor_m'])[:25],'interpretation':'Actual registered ID/barycentric anchors, with explicit source-ID to detailed-row map. One-hot weight alternatives are diagnostics only; they are not geometric correspondences. Indirect mechanical-to-source offsets are retained, not concealed. Geometry or physics admission not claimed.','accepted_physical_steps':0}
(P/'anchor-records.json').write_text(json.dumps(records,indent=2)+'\n');(P/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='wrong_min_onehot_examples'},indent=2))
