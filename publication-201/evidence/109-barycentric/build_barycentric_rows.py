#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy import sparse
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638'
D=S/'per-bank-workers-20261027/b109-114/attempt-018-barycentric-bone-witness-001'
BASE=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz'
CAND=S/'per-bank-workers-20261027/b109-114/attempt-016-self-barrier-union-001/backtracks/alpha-0p25/candidate.npz'
BASE130=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-130/summary.json'
BASE159=ROOT/'D/publication-182-stage-001/combined-653/audit-159/report.json'
ANALYSIS=S/'per-bank-workers-20261027/b109-114/attempt-017-bone-witness-constraints-001/bone-separation-analysis.json'
BANK=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight')
UNION=S/'per-bank-workers-20261027/b109-114/attempt-016-self-barrier-union-001/witness-pairs-union.npy'
ANALYSIS_SHA='353ab76f96aee0e5e84e0d338afbcdfb18d448c5d52b0a1651be725ff3e69388'
BODY_SHA='dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90'
BASE_SHA='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
CAND_SHA='e0a8d6314f051e001d87cbe254caf9f3ba7e45427de661a8c85e2da392ace3c4'
UNION_SHA='7f461cb82f9d972cfba9dc3bcc46867cb9ff2f38a3533ddf298efa8627887815'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if (D/'barycentric-bone-rows.json').exists():raise FileExistsError(D/'barycentric-bone-rows.json')
D.mkdir(parents=True,exist_ok=True)
if sha(BASE)!=BASE_SHA or sha(CAND)!=CAND_SHA or sha(ANALYSIS)!=ANALYSIS_SHA or sha(UNION)!=UNION_SHA:raise RuntimeError('input pin mismatch')
with np.load(BASE,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
with np.load(CAND,allow_pickle=False) as z:cand={k:z[k].copy() for k in z.files}
P=np.asarray(base['points_world_m'],dtype=np.float32);Q=np.asarray(cand['points_world_m'],dtype=np.float32);F=np.asarray(base['source_O_faces'],dtype=np.int64)
if not np.array_equal(F,cand['source_O_faces']) or set(base)!=set(cand):raise RuntimeError('topology/schema mismatch')
for k in base:
 if k!='points_world_m' and (base[k].dtype!=cand[k].dtype or base[k].shape!=cand[k].shape or base[k].tobytes()!=cand[k].tobytes()):raise RuntimeError('candidate ancestry/metadata mismatch '+k)
analysis=json.loads(ANALYSIS.read_text())
if analysis.get('status')!='ROWS_READY' or analysis.get('target_bank')!=[109,114] or len(analysis.get('pairs',[]))!=15:raise RuntimeError('source closest-feature analysis malformed')
selfpairs=np.asarray(np.load(UNION,allow_pickle=False),dtype=np.int64).reshape(-1,2)
target=[r for r in json.loads(BASE130.read_text())['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id']))==(109,114)]
targets=np.asarray(sorted({int(r['query_row']) for r in target}),dtype=np.int64)
edge=np.unique(np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
adj=sparse.csr_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(P),len(P)))
moving=set(map(int,targets));front=set(moving)
for _ in range(8):
 front=set(map(int,adj[list(front)].indices))-moving;moving.update(front)
support=set(moving);support.update(map(int,F[selfpairs.ravel()].ravel()))
support_rows=np.asarray(sorted(support),dtype=np.int64)
body_path=BANK/'exact-emitted.bodies.csv'
if sha(body_path)!=BODY_SHA:raise RuntimeError('pinned body-transform table hash mismatch')
with body_path.open(newline='') as f:
 bodies={int(r['body_index']):r for r in csv.DictReader(f)}
r=bodies[109]
M=np.asarray([[float(r[f'world_to_body_m{i}{j}']) for j in range(3)] for i in range(3)],dtype=np.float64)
rows=[];records=[];outside=[]
for info in analysis['pairs']:
 fi,bi=map(int,info['pair'])
 face_nodes=F[fi]
 w=np.asarray(info['baseline_skin_barycentric'],dtype=np.float64)
 ps,pb=[np.asarray(p,dtype=np.float64) for p in info['baseline_closest_points_mm']]
 dist=float(info['baseline_closest_distance_mm'])
 if abs(float(np.linalg.norm(ps-pb))-dist)>1e-8:raise RuntimeError(f'closest witness distance mismatch {fi}/{bi}')
 nlocal=(ps-pb)/dist
 nw=M.T@nlocal
 nw=nw/np.linalg.norm(nw)
 delta=np.asarray((Q[face_nodes].astype(np.float64)-P[face_nodes].astype(np.float64))*1000.0)
 witness_move=float((w@delta)@nw)
 frac=.25
 lower=-(1-frac)*dist
 coeffs=[];nodes=[]
 for node,weight in zip(face_nodes,w):
  node=int(node)
  if weight>1e-14 and node in support:
   nodes.append(node);coeffs.append((float(weight)*nw).tolist())
  elif weight>1e-14:
   outside.append({'pair':[fi,bi],'node':node,'weight':float(weight),'fixed_contribution':0.0})
 if not nodes:raise RuntimeError(f'no movable barycentric support for {fi}/{bi}')
 # Candidate motion along the exact closest-feature normal, compared with
 # the row lower bound. The bone witness point is fixed.
 records.append({'pair':[fi,bi],'skin_face_nodes':face_nodes.tolist(),'bone_triangle_id':bi,
  'baseline_distance_mm':dist,'skin_barycentric':w.tolist(),
  'bone_barycentric':info['baseline_bone_barycentric'],
  'closest_points_local_mm':[ps.tolist(),pb.tolist()],
  'normal_local_bone_to_skin':nlocal.tolist(),'normal_world_bone_to_skin':nw.tolist(),
  'required_delta_lower_mm':lower,'candidate_alpha025_witness_delta_mm':witness_move,
  'candidate_alpha025_row_slack_mm':witness_move-lower,
  'candidate_alpha025_satisfies_row':bool(witness_move>=lower-1e-9),
  'constraint_nodes':nodes,'constraint_coefficients':coeffs})
 rows.append({'nodes':nodes,'coefficients':coeffs,'lower':lower,'label':f'bone_closest:{fi}:{bi}'})
report={'schema':'numi.human.b109-114.closest-feature-barrier-rows.v1','status':'ROWS_READY',
 'baseline_candidate_sha256':BASE_SHA,'alpha025_candidate_sha256':CAND_SHA,
 'closest_feature_source_path':str(ANALYSIS),'closest_feature_source_sha256':ANALYSIS_SHA,
 'typed_body_transform_path':str(body_path),'typed_body_transform_sha256':BODY_SHA,'target_bank':[109,114],
 'self_witness_union_sha256':UNION_SHA,'self_witness_pair_count':int(len(selfpairs)),
 'support_rings':8,'moving_support_nodes':int(len(moving)),'total_support_nodes':int(len(support_rows)),
 'pair_count':len(records),'row_count':len(rows),'margin_fraction':.25,
 'constraint_semantics':'For each exact baseline closest skin/bone feature pair, constrain only the barycentric skin witness displacement along its bone-to-skin closest normal to retain 25 percent of the initial feature distance; bone remains fixed.',
 'pairs':records,'fixed_barycentric_contributions':outside,
 'candidate_direction_summary':{'rows_already_satisfied_by_alpha025':sum(x['candidate_alpha025_satisfies_row'] for x in records),
  'rows_violated_by_alpha025':sum(not x['candidate_alpha025_satisfies_row'] for x in records),
  'minimum_alpha025_row_slack_mm':min(x['candidate_alpha025_row_slack_mm'] for x in records),
  'maximum_alpha025_row_slack_mm':max(x['candidate_alpha025_row_slack_mm'] for x in records)}}
(D/'barycentric-bone-rows.json').write_text(json.dumps({'rows':rows},indent=2,sort_keys=True,allow_nan=False)+'\n')
(D/'barycentric-bone-analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'row_count':len(rows),'support_nodes':len(support_rows),
 'rows_already_satisfied_by_alpha025':report['candidate_direction_summary']['rows_already_satisfied_by_alpha025'],
 'rows_violated_by_alpha025':report['candidate_direction_summary']['rows_violated_by_alpha025'],
 'minimum_alpha025_row_slack_mm':report['candidate_direction_summary']['minimum_alpha025_row_slack_mm'],
 'maximum_alpha025_row_slack_mm':report['candidate_direction_summary']['maximum_alpha025_row_slack_mm'],
 'rows_path':str(D/'barycentric-bone-rows.json'),'analysis_path':str(D/'barycentric-bone-analysis.json')},indent=2),flush=True)
