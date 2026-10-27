from pathlib import Path
import hashlib, json
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027'
A69=ROOT/'D/publication-200-stage-001/evidence/retained/candidate.npz'
D071=S/'root-topology-a69-safe-subset-001/candidate.npz'
B109=S/'b109-114/attempt-024-a69-layer-fit-backtrack-001/alpha-0p015625/candidate.npz'
B61=S/'b61-97/root-layer-guard-004/candidate-private.npz'
OUT=Path(__file__).resolve().parent
CAND=OUT/'candidate.npz'
A69_SHA='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db'
D071_SHA='d071f04fa64ce602f08dbf7859c45b4518ef2032ac92af0c1100d69dbdd59826'
B109_SHA='468254c2d44472bbcaf91b86f968536477fa2847b8581bc69003c3eb2395e0e7'
B61_SHA='c65ba99363fe98d403d2ce2bbf6fea2eff89b6ae1758bcd40e7274e6a2b9f8ed'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rows_changed(a,b):
 if a.dtype!=np.float32 or b.dtype!=np.float32 or a.shape!=b.shape:raise ValueError('position array dtype/shape mismatch')
 return np.flatnonzero(np.any(a.view(np.uint32)!=b.view(np.uint32),axis=1))

for path,expect,name in ((A69,A69_SHA,'a69'),(D071,D071_SHA,'d071'),(B109,B109_SHA,'b109-114'),(B61,B61_SHA,'b61-97')):
 got=sha(path)
 if got!=expect:raise RuntimeError(f'{name} hash mismatch {got}')
a69=load(A69);d071=load(D071);b109=load(B109);b61=load(B61)
if CAND.exists():raise FileExistsError(CAND)
if not (a69.keys()==b109.keys()==b61.keys()):raise RuntimeError('position patch schemas differ from a69')
if not (a69.keys() <= d071.keys()):raise RuntimeError('d071 lacks source candidate keys')
# Patch candidates are coordinate-only. Require all other bytes to equal a69.
for label,z in (('b109',b109),('b61',b61)):
 for k in a69:
  if k=='points_world_m':continue
  if z[k].dtype!=a69[k].dtype or z[k].shape!=a69[k].shape or z[k].tobytes()!=a69[k].tobytes():
   raise RuntimeError(f'{label} has noncoordinate metadata divergence at {k}')
A=np.asarray(a69['points_world_m'],dtype=np.float32)
D=np.asarray(d071['points_world_m'],dtype=np.float32)
P109=np.asarray(b109['points_world_m'],dtype=np.float32)
P61=np.asarray(b61['points_world_m'],dtype=np.float32)
if A.shape!=D.shape or not np.array_equal(A.view(np.uint32),D.view(np.uint32)):
 raise RuntimeError('d071 point rows do not match a69; position deltas cannot be applied unchanged')
for label,z in (('b109',b109),('b61',b61)):
 if z['points_world_m'].shape!=A.shape or not np.isfinite(z['points_world_m']).all():raise RuntimeError(f'{label} invalid position array')
 if z['source_O_faces'].dtype!=a69['source_O_faces'].dtype or not np.array_equal(z['source_O_faces'],a69['source_O_faces']):raise RuntimeError(f'{label} topology differs from a69')
i109=rows_changed(A,P109);i61=rows_changed(A,P61)
overlap=np.intersect1d(i109,i61)
if len(overlap):raise RuntimeError(f'coordinate supports overlap: {overlap[:64].tolist()}')
# Verify patches contain no hidden change outside the reported supports.
if not np.array_equal(P109[np.setdiff1d(np.arange(len(A)),i109)].view(np.uint32),A[np.setdiff1d(np.arange(len(A)),i109)].view(np.uint32)):raise RuntimeError('b109 has a hidden off-support coordinate change')
if not np.array_equal(P61[np.setdiff1d(np.arange(len(A)),i61)].view(np.uint32),A[np.setdiff1d(np.arange(len(A)),i61)].view(np.uint32)):raise RuntimeError('b61 has a hidden off-support coordinate change')
Y=D.copy();Y[i109]=P109[i109];Y[i61]=P61[i61]
C={k:v.copy() for k,v in d071.items()};C['points_world_m']=Y
# Ensure every d071 nonposition metadata byte is preserved.
for k in d071:
 if k!='points_world_m' and (C[k].dtype!=d071[k].dtype or C[k].shape!=d071[k].shape or C[k].tobytes()!=d071[k].tobytes()):raise RuntimeError('d071 metadata changed during combine: '+k)
with CAND.open('xb') as f:np.savez_compressed(f,**C)
actual=rows_changed(D,Y)
if not np.array_equal(actual,np.sort(np.concatenate((i109,i61)))):raise RuntimeError('merged changed-node set mismatch')
np.savez_compressed(OUT/'patch-nodes.npz',
 b109_node_rows=i109.astype(np.int64),b61_node_rows=i61.astype(np.int64),
 union_node_rows=np.sort(np.concatenate((i109,i61))).astype(np.int64),
 d071_points_world_m_f32=D[np.sort(np.concatenate((i109,i61)))],
 merged_points_world_m_f32=Y[np.sort(np.concatenate((i109,i61)))])
changed_faces=np.flatnonzero(np.any(C['source_O_faces']!=a69['source_O_faces'],axis=1))
report={
 'schema':'numi.human.combined.d071.b109_114.b61_97.preflight.v1',
 'status':'DISJOINT_COORDINATE_PATCHES_COMBINED_ON_D071',
 'input_pins':{'a69':{'path':str(A69),'sha256':sha(A69)},'d071':{'path':str(D071),'sha256':sha(D071)},'b109_114_468':{'path':str(B109),'sha256':sha(B109)},'b61_97_c65':{'path':str(B61),'sha256':sha(B61)}},
 'compatibility':{'node_rows_same':True,'d071_positions_bitwise_equal_a69':True,'position_patch_schemas_equal_a69':True,'position_patch_noncoordinate_arrays_bytewise_equal_a69':True,'d071_all_nonposition_arrays_preserved_bytewise':True,'patch_topology_matches_a69':True,'overlap_rows':overlap.astype(int).tolist(),'metadata_rule':'Preserve every d071 array exactly except points_world_m; apply each position patch only on its disjoint rows.'},
 'support':{'b109_changed_node_count':int(len(i109)),'b61_changed_node_count':int(len(i61)),'changed_node_overlap_count':int(len(overlap)),'union_changed_node_count':int(len(actual)),'b109_rows_sha256':hashlib.sha256(i109.astype(np.int64).tobytes()).hexdigest(),'b61_rows_sha256':hashlib.sha256(i61.astype(np.int64).tobytes()).hexdigest(),'max_movement_from_d071_mm':float(np.linalg.norm(Y.astype(np.float64)-D.astype(np.float64),axis=1).max()*1000)},
 'd071_connectivity':{'changed_face_rows_vs_a69':int(len(changed_faces)),'changed_face_rows':changed_faces.astype(int).tolist(),'faces_preserved_from_d071':True},
 'candidate':{'path':str(CAND),'sha256':sha(CAND),'points':int(len(Y)),'faces':int(len(C['source_O_faces'])),'node_patch_path':str(OUT/'patch-nodes.npz'),'node_patch_sha256':sha(OUT/'patch-nodes.npz')},
 'geometry_admitted':False,'physical_steps':0}
(OUT/'preflight.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'candidate_sha256':report['candidate']['sha256'],'b109_nodes':len(i109),'b61_nodes':len(i61),'overlap':len(overlap),'union':len(actual),'changed_d071_faces':len(changed_faces),'preflight_sha256':sha(OUT/'preflight.json')},indent=2))
