#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638'
D=S/'per-bank-workers-20261027/b61-97/root-layer-guard-004'
BASE=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz'
CAND=S/'per-bank-workers-20261027/b61-97/root-layer-guard-002/backtrack-0p6/candidate-private.npz'
BASE159=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/audit-159/report.json'
BASE130=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/audit-130/summary.json'
BANKHELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
BANK=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
SELF=Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py')
UNION=S/'per-bank-workers-20261027/b61-97/root-layer-guard-001/audit-130/surface-self-pairs.npy'
EXPECTED_BASE='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db'
EXPECTED_UNION='2640141f6089c80ec9120bc4a2b8aed4bd5de4cc0303f83bd3897abed5fd26e4'
ADDED=[(8838,41),(8839,41),(8841,41),(8849,41)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p)
 if sp is None or sp.loader is None:raise RuntimeError(f'cannot import {p}')
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
if (D/'bone-separation-analysis.json').exists():raise FileExistsError(D/'bone-separation-analysis.json')
D.mkdir(parents=True,exist_ok=True)
if sha(BASE)!=EXPECTED_BASE or sha(UNION)!=EXPECTED_UNION:raise RuntimeError('baseline or union witness pin mismatch')
with np.load(BASE,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
with np.load(CAND,allow_pickle=False) as z:cand={k:z[k].copy() for k in z.files}
P=np.asarray(base['points_world_m'],dtype=np.float32); Q=np.asarray(cand['points_world_m'],dtype=np.float32); F=np.asarray(base['source_O_faces'],dtype=np.int64)
if set(base)!=set(cand) or not np.array_equal(F,cand['source_O_faces']):raise RuntimeError('typed candidate schema/topology mismatch')
for k in base:
 if k!='points_world_m' and (base[k].dtype!=cand[k].dtype or base[k].shape!=cand[k].shape or base[k].tobytes()!=cand[k].tobytes()):raise RuntimeError('nonposition metadata mismatch '+k)
baseline=json.loads(BASE159.read_text())
bankrec=next(b for b in baseline['banks'] if (int(b['body']),int(b['stable_bone_id']))==(61,97))
baseline_pairs={tuple(map(int,p)) for p in (bankrec.get('result') or {}).get('triangle_pairs',[])}
if len(baseline_pairs)!=32 or baseline_pairs.intersection(ADDED):raise RuntimeError('added pair set does not match pinned 32-pair baseline')
helper=mod('bone_witness_helper',BANKHELPER)
T,ids,gids,bodies,M,offset,pos,keys,pins,unused=helper.read_bank_inputs(BANK,E)
body=61;bone_id=97
mask=(ids[:,0]==body)&(ids[:,1]==bone_id)
bone=T[mask].astype(np.float64)*1000.0
bone_ids=ids[mask,3].astype(np.int64)
if len(np.unique(bone_ids))!=len(bone_ids):raise RuntimeError('bone face identity is not unique')
loc0=((P.astype(np.float64)-pos[body])@M[body].T-offset[body])*1000.0
loc1=((Q.astype(np.float64)-pos[body])@M[body].T-offset[body])*1000.0
sep=mod('bone_separation_source',SELF)
target_rows=[r for r in json.loads(BASE130.read_text())['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id']))==(61,97)]
targets=np.asarray([20993],dtype=np.int64)
pairs=np.asarray(np.load(UNION,allow_pickle=False),dtype=np.int64).reshape(-1,2)
edge=np.unique(np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
from scipy import sparse
adj=sparse.csr_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(P),len(P)))
support=set(map(int,targets));frontier=set(support)
for _ in range(4):
 frontier=set(map(int,adj[list(frontier)].indices))-support; support.update(frontier)
support_moving=set(support);support.update(map(int,F[pairs.ravel()].ravel())); support=np.asarray(sorted(support),dtype=np.int64)
records=[];rows=[];failed=[]
for si,bi in ADDED:
 hits=np.flatnonzero(bone_ids==bi)
 if len(hits)!=1:raise RuntimeError(f'bone triangle identity {bi} maps to {len(hits)} rows')
 fi=int(si);pa=loc0[F[fi]];pc=loc1[F[fi]];pb=bone[int(hits[0])]
 dist,qa,qb,ba,bb=sep._closest_triangles(pa,pb)
 opts=[]
 for axis in sep._sat_axes(pa,pb):
  axis=np.asarray(axis,dtype=np.float64)
  g=float(np.min(pa@axis)-np.max(pb@axis))
  n=axis
  if g<=0:
   n=-axis;g=float(np.min(pa@n)-np.max(pb@n))
  if g>1e-10:opts.append((g,n))
 if not opts:
  failed.append({'pair':[fi,bi],'closest_distance_mm':float(dist),'reason':'baseline has no strictly positive SAT separation axis'})
  continue
 gap,n=max(opts,key=lambda x:x[0])
 margin=.25*gap
 bone_hi=float(np.max(pb@n))
 world_n=M[body].T@n
 world_n=world_n/np.linalg.norm(world_n)
 base_proj=pa@n; cand_proj=pc@n
 clearance=base_proj-bone_hi
 step=(pc-pa)@n
 entries=[]
 for k,node in enumerate(F[fi]):
  node=int(node); lower=float(bone_hi+margin-base_proj[k])
  entries.append({'node':node,'base_clearance_mm':float(clearance[k]),'required_delta_lower_mm':lower,
   'candidate_projected_delta_mm':float(step[k]),'candidate_clearance_mm':float(cand_proj[k]-bone_hi-margin),
   'in_support':bool(node in set(map(int,support_moving)))})
  if node in set(map(int,support_moving)):
   rows.append({'nodes':[node],'coefficients':[world_n.tolist()],'lower':lower,'label':f'bone:{fi}:{bi}:v{node}'})
  elif lower>1e-9:
   failed.append({'pair':[fi,bi],'node':node,'reason':'required movable separation node is fixed outside support','lower_mm':lower})
 records.append({'pair':[fi,bi],'bone_triangle_row':int(hits[0]),'bone_triangle_stable_id':int(bi),
  'baseline_closest_distance_mm':float(dist),'baseline_skin_barycentric':ba.tolist(),'baseline_bone_barycentric':bb.tolist(),
  'baseline_closest_points_mm':[qa.tolist(),qb.tolist()],'separating_axis_local':n.tolist(),'separating_axis_world':world_n.tolist(),
  'baseline_projected_gap_mm':gap,'preserved_margin_mm':margin,'vertex_rows':entries,
  'baseline_pair_absent':(fi,bi) not in baseline_pairs})
report={'schema':'numi.human.b61-97.targeted-bone-separation-witnesses.v1','status':'ROWS_READY' if not failed else 'OBSTRUCTION',
'baseline_candidate_sha256':sha(BASE),'alpha060_candidate_sha256':sha(CAND),'alpha060_candidate_points_sha256':hashlib.sha256(Q.tobytes()).hexdigest(),
'baseline_audit159_sha256':sha(BASE159),'target_bank':[61,97],'baseline_comparable_pair_count':len(baseline_pairs),
'added_exact_pairs':[list(p) for p in ADDED],'added_pairs_absent_from_baseline':all((int(a),int(b)) not in baseline_pairs for a,b in ADDED),
'self_union_sha256':sha(UNION),'unique_self_witness_pair_count':int(len(pairs)),
'bank_input_pins':pins,'support_node_count':int(len(support)),'moving_support_node_count':int(len(support_moving)),
'separation_method':'Choose the largest strictly positive SAT projection gap for each nonintersecting baseline skin/bone triangle pair. Add one fixed-bone halfspace row per supported source-face vertex, retaining 25 percent of the baseline projection gap.',
'row_count':len(rows),'pairs':records,'obstructions':failed,
'intended_correction_projection_summary':{'positive_vertex_steps':sum(x['candidate_projected_delta_mm']>0 for r in records for x in r['vertex_rows']),
'negative_vertex_steps':sum(x['candidate_projected_delta_mm']<0 for r in records for x in r['vertex_rows']),
'minimum_candidate_clearance_to_preserved_margin_mm':min((x['candidate_clearance_mm'] for r in records for x in r['vertex_rows']),default=None),
'candidate_intersection_pairs_count':len(ADDED)}}
(D/'bone-separation-analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
(D/'bone-separation-rows.json').write_text(json.dumps({'rows':rows},indent=2,sort_keys=True,allow_nan=False)+'\n')
report['analysis_report_sha256']=sha(D/'bone-separation-analysis.json')
report['constraint_rows_sha256']=sha(D/'bone-separation-rows.json')
(D/'analysis-receipt.json').write_text(json.dumps({k:v for k,v in report.items() if k not in ('pairs','obstructions')},indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'baseline_separated_pairs':len(records),'row_count':len(rows),'obstructions':failed,
 'min_baseline_gap_mm':min((r['baseline_projected_gap_mm'] for r in records),default=None),
 'max_baseline_gap_mm':max((r['baseline_projected_gap_mm'] for r in records),default=None),
 'min_alpha060_clearance_mm':report['intended_correction_projection_summary']['minimum_candidate_clearance_to_preserved_margin_mm'],
 'positive_vertex_steps':report['intended_correction_projection_summary']['positive_vertex_steps'],
 'negative_vertex_steps':report['intended_correction_projection_summary']['negative_vertex_steps'],
 'analysis':str(D/'bone-separation-analysis.json'),'rows':str(D/'bone-separation-rows.json')},indent=2),flush=True)
