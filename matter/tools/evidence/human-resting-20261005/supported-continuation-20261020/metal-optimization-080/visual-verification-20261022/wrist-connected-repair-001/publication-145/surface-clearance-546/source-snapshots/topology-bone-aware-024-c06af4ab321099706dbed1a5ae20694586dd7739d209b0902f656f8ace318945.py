#!/usr/bin/env python3
"""Exact-pair-reducing fixed-coordinate connected flip proposal (construction only).
Uses current 159 crossing pairs, 402/405/541 flip gates, 370's exact replacement counter,
and 408's incremental 315-predicate self witness screen. Never runs a full final audit.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
D=ROOT/'D'
DEFAULT_CANDIDATE=D/'contained-clearance-545/last-exit-022/direct/candidate.npz'
DEFAULT_A159=D/'contained-clearance-545/last-exit-022/check/audit-159/report.json'
DEFAULT_A130=D/'contained-clearance-545/last-exit-022/check/audit-130/summary.json'
DEFAULT_P130=D/'contained-clearance-545/last-exit-022/check/audit-130/surface-self-pairs.npy'
OWNER370=D/'connected-remesh-reuse-370/select_bone_reducing_flips_370.py'
OWNER405=D/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
OWNER402=D/'local-two-parent-retriangulation-402/two_parent_patch_flips_402.py'
OWNER541=D/'focused-clearance-541/topology/repair_focused_clearance_541.py'
CHECK408=D/'local-two-parent-retriangulation-408/incremental_exact_flip_check_408.py'
SEARCH409=D/'local-two-parent-retriangulation-408/exact_global_incremental_flip_search_409.py'
EXPECTED={
 'candidate':'ced651c06f4363139288fc006ce9bdbf62ef13f603491c16fbdf0d98eb654bc4',
 'audit159':'dd1dc6d851391978dbfc8b18b4dfd6dfa8e6dd04e3d174f12f21d513cdf25e43',
 'audit130':'bdefd631243673d3cb2476bdef6c2a83a5ebc28b21d18111658c9e60e6d499db',
 'pairs130':'55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de',
 'owner370':'c8650b8c311ca8f386ebf1713b5fc285b7bded860cfa878192472704ac1336d5',
 'owner405':'e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967',
 'owner402':'f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764',
 'owner541':'26930be05518834f02724948cffa5ff74719f070a4ac4e4917b0965f42a412aa',
 'check408':'01026156cccd5c1742e2edcaa6288f64f888731424dfb6337831412701020711',
 'search409':'64f05be466d672266d4265185c895fecc70113d476f69e7a1c6dc3af8ec514a8',
 'exact315':'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d',
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def shaarr(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def identical(a,b):
 x,y=np.ascontiguousarray(a),np.ascontiguousarray(b)
 return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

def evaluate_op_local(q,c,op,quality_before,points_mm,ref_mm):
 rows=list(map(int,op['face_rows']));nf=np.asarray(op['new_faces'],dtype=np.int64)
 before={k:np.asarray(quality_before[k])[rows] for k in ('normal_cos','area_ratio','min_angle_deg')}
 after=q.face_quality(points_mm,ref_mm,nf,nf)
 old_ratio=float(before['area_ratio'].min());new_ratio=float(np.asarray(after['area_ratio']).min())
 old_cos=float(before['normal_cos'].min());new_cos=float(np.asarray(after['normal_cos']).min())
 old_ang=float(before['min_angle_deg'].min());new_ang=float(np.asarray(after['min_angle_deg']).min())
 if np.any(after['area_ratio']<q.MIN_AREA_RATIO):return None,'fixed037_area_floor'
 if np.any(after['normal_cos']<=0):return None,'new_face_not_aligned_to_fixed037'
 if new_ang<q.MIN_PATCH_ANGLE_DEG or new_ang-old_ang < -q.MAX_MIN_ANGLE_LOSS_DEG-1e-10:return None,'angle_loss_bound'
 delta=(new_ratio-old_ratio,new_cos-old_cos,new_ang-old_ang)
 if not (delta[0]>=q.MIN_RATIO_IMPROVEMENT or delta[1]>=q.MIN_COS_IMPROVEMENT or delta[2]>=q.MIN_ANGLE_IMPROVEMENT_DEG):return None,'no_measured_quality_gain'
 return {'score':delta,'before':{'minimum_area_ratio':old_ratio,'minimum_normal_cosine':old_cos,'minimum_angle_deg':old_ang},'after':{'minimum_area_ratio':new_ratio,'minimum_normal_cosine':new_cos,'minimum_angle_deg':new_ang},'deltas':{'area_ratio':delta[0],'normal_cosine':delta[1],'min_angle_deg':delta[2]}},'eligible'

def csr_rows(ptr,values,n):
 ptr=np.asarray(ptr);values=np.asarray(values)
 if ptr.shape!=(n+1,) or int(ptr[0])!=0 or int(ptr[-1])!=len(values) or np.any(ptr[1:]<ptr[:-1]):raise ValueError('invalid CSR')
 return [list(map(int,values[int(ptr[i]):int(ptr[i+1])])) for i in range(n)]

def detail_rows(c,patch=None):
 n=len(c['source_O_faces'])
 if 'face_fine_detail_ancestry_indptr' in c:return csr_rows(c['face_fine_detail_ancestry_indptr'],c['face_fine_detail_ancestry_rows'],n)
 src=csr_rows(c['face_source_ancestry_indptr'],c['face_source_ancestry_face_rows'],n)
 return [[int(d)] if int(d)>=0 else sorted(set(src[i])&set(patch or ())) for i,d in enumerate(c['parent_fine_detail_face_row'])]

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--candidate',type=Path,default=DEFAULT_CANDIDATE)
 ap.add_argument('--audit159',type=Path,default=DEFAULT_A159)
 ap.add_argument('--audit130',type=Path,default=DEFAULT_A130)
 ap.add_argument('--self-pairs130',type=Path,default=DEFAULT_P130)
 ap.add_argument('--expected-candidate-sha256',required=True)
 ap.add_argument('--expected-audit159-sha256',required=True)
 ap.add_argument('--expected-audit130-sha256',required=True)
 ap.add_argument('--expected-self-pairs-sha256',required=True)
 ap.add_argument('--output-dir',type=Path,required=True)
 ap.add_argument('--max-candidates',type=int,default=136)
 ap.add_argument('--max-accept',type=int,default=64)
 ap.add_argument('--prefilter-factor',type=int,default=4,help='validate at most factor*max-candidates edges after ranking by current exact pair incidence')
 ap.add_argument('--progress-every',type=int,default=8)
 ap.add_argument('--exclude-self-pairs',type=Path,default=None,help='optional exact 130 self-pair NPY from a previous proposal audit; any operation touching these face rows is replay-filtered')
 ap.add_argument('--expected-exclusion-pairs-sha256',default=None)
 args=ap.parse_args()
 base_path=args.candidate.resolve();outdir=args.output_dir.resolve()
 if outdir.exists():raise FileExistsError(outdir)
 if args.max_candidates<1 or args.max_accept<1 or args.prefilter_factor<1 or args.progress_every<1:raise ValueError('limits must be positive')
 outdir.mkdir(parents=True,exist_ok=False)
 progress_path=outdir/'progress.jsonl';started=time.perf_counter()
 def progress(stage,**fields):
  rec={'stage':stage,'elapsed_seconds':round(time.perf_counter()-started,3),**fields}
  with progress_path.open('a') as pf:pf.write(json.dumps(rec,sort_keys=True)+'\n');pf.flush()
  print(json.dumps(rec,sort_keys=True),flush=True)
 progress('preflight_started')
 input_pins={'candidate':sha(base_path),'audit159':sha(args.audit159),'audit130':sha(args.audit130),'self_pairs130':sha(args.self_pairs130)}
 for k,expected in [('candidate',args.expected_candidate_sha256),('audit159',args.expected_audit159_sha256),('audit130',args.expected_audit130_sha256),('self_pairs130',args.expected_self_pairs_sha256)]:
  if input_pins[k]!=expected:raise ValueError(f'{k} SHA pin mismatch: {input_pins[k]}')
 owners={'owner370':OWNER370,'owner405':OWNER405,'owner402':OWNER402,'owner541':OWNER541,'check408':CHECK408,'search409':SEARCH409}
 owner_pins={k:sha(v) for k,v in owners.items()}
 for k,v in owner_pins.items():
  if v!=EXPECTED[k]:raise ValueError(f'{k} source pin mismatch: {v}')
 base=load_npz(base_path);F=np.asarray(base['source_O_faces'])
 audit159=json.loads(args.audit159.read_text());audit130=json.loads(args.audit130.read_text())
 if audit159.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT' or audit159.get('complete') is not True or audit159.get('bank_count')!=185 or audit159.get('expected_bank_count')!=185 or audit159.get('candidate_sha256')!=input_pins['candidate']:raise ValueError('audit159 is incomplete or not bound to this candidate')
 if audit130.get('complete_surface_audit') is not True or audit130.get('candidate_sha256')!=input_pins['candidate'] or int(audit130.get('exact_PLC_surface_pairs',-1))!=0 or audit130.get('surface_predicate_sha256')!=EXPECTED['exact315']:raise ValueError('baseline 130 is not complete/self-free or predicate mismatches')
 baseline_self=np.load(args.self_pairs130,allow_pickle=False)
 if baseline_self.shape!=(0,2):raise ValueError('baseline self-pair file is not empty [0,2]')
 exclude_pairs=np.empty((0,2),dtype=np.int64)
 if args.exclude_self_pairs:
  if not args.expected_exclusion_pairs_sha256:raise ValueError('exclusion self-pair SHA is required')
  if sha(args.exclude_self_pairs)!=args.expected_exclusion_pairs_sha256:raise ValueError('exclusion self-pair SHA mismatch')
  exclude_pairs=np.load(args.exclude_self_pairs,allow_pickle=False)
  if exclude_pairs.ndim!=2 or exclude_pairs.shape[1]!=2 or not np.issubdtype(exclude_pairs.dtype,np.integer):raise ValueError('exclusion pairs must be integer Nx2')
  if np.any(exclude_pairs<0) or np.any(exclude_pairs>=len(F)):raise ValueError('exclusion witness face out of range')
 excluded_faces=set(map(int,exclude_pairs.reshape(-1)))
 # Load the existing all-185 exact counter; do not invoke its 358 topology proposal path.
 sel=loadmod('select_bone_reducing_flips_370_counter023',OWNER370)
 flip, surface_owner, bank_helper, exact159, runtime_hashes=sel.load_pinned_runtime()
 raw_inputs=bank_helper.read_bank_inputs(sel.BANK_DIR_DEFAULT,sel.EVIDENCE_ROOT_DEFAULT)
 banks,bank_keys=sel.typed_banks_from_159_inputs(raw_inputs)
 bank_helper.summarize_surface_crossing_report(audit159,expected_banks=185,expected_bank_keys=bank_keys)
 pairs_by_bank,target_rows,base_bank_counts=sel._pair_rows_and_counts(audit159,candidate_sha256=input_pins['candidate'],face_count=len(F),banks=banks,expected_bank_keys=bank_keys)
 if not target_rows:raise ValueError('fresh 159 report has no crossing target faces')
 face_base_counts={k:Counter(int(fr) for fr,_tri in ps) for k,ps in pairs_by_bank.items()}
 counter=sel.Exact159PatchCounter(base['points_world_m'],banks,exact159,surface_owner.records)
 # Current topology is 541-007's face + full CSR topology, while coordinates are current 022.
 mod405=loadmod('cross_seam_search_405_boneaware023',OWNER405);mod405.OWNER402=OWNER402
 detail=detail_rows(base);detail_ids=sorted({d for r in target_rows for d in detail[r]})
 if not detail_ids:raise ValueError('crossing seed rows lack typed detail ancestry')
 detail_set=set(detail_ids)
 def group_of(ds):
  s=set(map(int,ds));return 0 if s and s.issubset(detail_set) else None
 mod405.DETAIL_GROUPS=(tuple(detail_ids),);mod405.detail_csr=detail_rows;mod405.group_of=group_of
 owner402=loadmod('two_parent_patch_flips_402_boneaware023',OWNER402);owner402.validate_candidate(base)
 # 405 reloads 402 and reconstructs the full edge map for each candidate. Pin one
 # in-memory 402 module and cache edge maps by the immutable face-array identity.
 raw_edge_map=owner402.edge_map;edge_map_cache={}
 def cached_edge_map(f):
  key=id(f);item=edge_map_cache.get(key)
  if item is not None and item[0] is f:return item[1]
  value=raw_edge_map(f);edge_map_cache[key]=(f,value);return value
 owner402.edge_map=cached_edge_map
 raw405_loader=mod405.load_mod
 def cached405_loader(name,path):
  if Path(path).resolve()==OWNER402.resolve():return owner402
  return raw405_loader(name,path)
 mod405.load_mod=cached405_loader
 raw_csr_rows=mod405.csr_rows;csr_cache={}
 def cached_csr_rows(ptr,values,n):
  key=(id(ptr),id(values),int(n));item=csr_cache.get(key)
  if item is not None and item[0] is ptr and item[1] is values:return item[2]
  rows=raw_csr_rows(ptr,values,n);csr_cache[key]=(ptr,values,rows);return rows
 mod405.csr_rows=cached_csr_rows
 if not np.array_equal(F,base['area_reference_037_source_O_faces']):raise ValueError('current/037 topology mismatch')
 anchors=np.asarray(base['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
 if anchors.shape!=(237,) or len(np.unique(anchors))!=237:raise ValueError('typed anchor array mismatch')
 anchor_xyz=base['points_world_m'][anchors].copy();points_before=base['points_world_m'].copy()
 locks0,outer0,between0,_=mod405.boundary_locks(base,detail,detail_set)
 q=loadmod('repair_focused_clearance_541_boneaware023',OWNER541)
 points_mm=np.asarray(base['points_world_m'],dtype=np.float64)*1000.0
 ref_mm=np.asarray(base['area_reference_037_points_world_m_f64'],dtype=np.float64)*1000.0
 quality=q.face_quality(points_mm,ref_mm,F,base['area_reference_037_source_O_faces'])
 # Prepare the cached exact 130 surface records using the already-complete zero-pair receipt; do not rerun a full self audit.
 search=loadmod('exact_global_incremental_flip_search_409_boneaware023',SEARCH409);checker=loadmod('incremental_exact_flip_check_408_boneaware023',CHECK408)
 pred=search.PRED;pred_sha=sha(pred)
 if pred_sha!=EXPECTED['exact315']:raise ValueError('408 incremental checker predicate differs from exact 315')
 exact130=search.exact_owner()
 records,_=exact130.float32_surface_records(np.asarray(base['points_world_m'],dtype=np.float32),F,source_seam_groups=())
 records=exact130._immutable_self_pair_records(records)
 index=exact130._prepare_surface_aabb(records)
 state_points=np.array(base['points_world_m'],copy=True);state_faces=np.array(F,copy=True);state_points.setflags(write=False);state_faces.setflags(write=False)
 state=checker.State(search.exact_owner(),state_points,state_faces,records,index,frozenset(),(),input_pins['candidate'],pred_sha)
 # Cheaply rank target-touching interior edges before invoking source/quality validators.
 em=owner402.edge_map(F);seedset=set(map(int,target_rows));proposals=[];rejected=Counter()
 cheap=[]
 for edge,inc in em.items():
  if len(inc)!=2:continue
  rows=tuple(sorted(int(v[0]) for v in inc))
  if not seedset.intersection(rows):continue
  old_total=sum(face_base_counts[k].get(r,0) for k in bank_keys for r in rows)
  if old_total<=0:continue
  cheap.append((old_total,tuple(sorted(map(int,edge))),inc,rows))
 cheap.sort(key=lambda z:(-z[0],z[1],z[3]))
 prefilter_limit=args.prefilter_factor*args.max_candidates
 selected_cheap=cheap[:prefilter_limit]
 progress('cheap_prefilter_complete',target_faces=len(seedset),interior_target_edges=len(cheap),validated_edge_cap=prefilter_limit,selected_for_validation=len(selected_cheap),top_old_pair_count=selected_cheap[0][0] if selected_cheap else 0)
 # 405 repeatedly asks for the same full CSR ancestry during a base scan. Cache by object identity.
 detail_cache={}
 def cached_detail_rows(c,patch=None):
  key=id(c);item=detail_cache.get(key)
  if item is not None and item[0] is c:return item[1]
  value=detail_rows(c,patch);detail_cache[key]=(c,value);return value
 mod405.detail_csr=cached_detail_rows
 fast_parity_checked=False
 for ordinal,(old_total,edge,inc,cheap_rows) in enumerate(selected_cheap,1):
  if ordinal==1 or ordinal%args.progress_every==0:progress('source_quality_validation',validated=ordinal,total= len(selected_cheap),eligible=len(proposals),rejections=dict(rejected))
  op,why=mod405.propose_one(base,edge,inc,locks0)
  if op is None:rejected[why]+=1;continue
  rows=list(map(int,op['face_rows']))
  if excluded_faces.intersection(rows):rejected['touches_explicit_self_witness']+=1;continue
  qual,why=evaluate_op_local(q,base,op,quality,points_mm,ref_mm)
  if not fast_parity_checked:
   slow,slow_why=q.evaluate_op(base,op,quality,points_mm,ref_mm)
   if slow_why!=why or (slow is None)!=(qual is None):raise ValueError('local 541 evaluator parity failed')
   if qual is not None and (slow['score']!=qual['score'] or slow['deltas']!=qual['deltas']):raise ValueError('local 541 evaluator numeric parity failed')
   fast_parity_checked=True
   progress('541_local_evaluator_parity',matched=True,first_eligible_edge=list(edge))
  if qual is None:rejected['541_'+why]+=1;continue
  old_by={k:sum(face_base_counts[k].get(r,0) for r in rows) for k in bank_keys}
  proposals.append({'edge':edge,'op':op,'quality':qual,'old_by_bank':old_by,'old_total':sum(old_by.values())})
 progress('source_quality_validation_complete',validated=len(selected_cheap),eligible=len(proposals),rejections=dict(rejected))
 # Score a bounded high-witness subset with exact 159 replacement triangles across all 185 banks.
 proposals.sort(key=lambda p:(-p['old_total'],-max(p['quality']['score']),p['edge'],tuple(p['op']['face_rows'])))
 scored=[]
 for score_index,p in enumerate(proposals[:args.max_candidates],1):
  if score_index==1 or score_index%args.progress_every==0:progress('exact_159_scoring',scored=score_index,total=min(len(proposals),args.max_candidates),reducing=len(scored))
  rows=list(map(int,p['op']['face_rows']));new_faces=np.asarray(p['op']['new_faces'],dtype=np.int64)
  new_result=counter.count_faces(rows,new_faces);new_by={k:int(new_result[k]['count']) for k in bank_keys}
  keep,why,reduction=sel._strict_decrease(p['old_by_bank'],new_by)
  p['new_by_bank']=new_by;p['reduction']=int(reduction);p['strict_bank_reduction']=bool(keep);p['pair_reason']=why;p['new_results']=new_result
  if keep:scored.append(p)
  else:rejected['exact159_'+why]+=1
 scored.sort(key=lambda p:(-p['reduction'],-max(p['quality']['score']),p['edge'],tuple(p['op']['face_rows'])))
 current={k:np.array(v,copy=True) for k,v in base.items()};accepted=[];used_faces=set();used_nodes=set();self_rejections=[]
 qcurrent=q.face_quality(points_mm,ref_mm,current['source_O_faces'],current['area_reference_037_source_O_faces'])
 for p in scored:
  if len(accepted)>=args.max_accept:break
  edge=p['edge'];emc=owner402.edge_map(current['source_O_faces']);inc=emc.get(edge,[])
  if len(inc)!=2:rejected['stale_edge_after_batch']+=1;continue
  op,why=mod405.propose_one(current,edge,inc,locks0)
  if op is None:rejected['stale_'+why]+=1;continue
  rows=list(map(int,op['face_rows']))
  quad=set(map(int,current['source_O_faces'][rows].reshape(-1)))
  if set(rows)&used_faces or quad&used_nodes:rejected['overlap_after_exact_ranking']+=1;continue
  qual,why=evaluate_op_local(q,current,op,qcurrent,points_mm,ref_mm)
  if qual is None:rejected['stale_541_'+why]+=1;continue
  # Exact 159 score must remain reproducible against original face rows.
  new_result=counter.count_faces(rows,np.asarray(op['new_faces'],dtype=np.int64))
  new_by={k:int(new_result[k]['count']) for k in bank_keys}
  old_by={k:sum(face_base_counts[k].get(r,0) for r in rows) for k in bank_keys}
  keep,why,reduction=sel._strict_decrease(old_by,new_by)
  if not keep:rejected['recheck_exact159_'+why]+=1;continue
  trial=mod405.apply_op(current,op)
  incremental=checker.evaluate(state,trial['points_world_m'],trial['source_O_faces'])
  if incremental.report.get('introduced_pairs'):
   self_rejections.append({'edge':list(edge),'face_rows':rows,'introduced_pairs':incremental.report['introduced_pairs']})
   rejected['incremental_self_witness']+=1;continue
  state=checker.commit(state,incremental,require_reduction=False,candidate_sha256=input_pins['candidate'])
  if not identical(trial['points_world_m'],points_before):raise ValueError('point coordinates changed')
  if not np.array_equal(trial['source_O_faces'],trial['area_reference_037_source_O_faces']):raise ValueError('fixed-037 face synchronization lost')
  dnow=detail_rows(trial);locks1,outer1,between1,_=mod405.boundary_locks(trial,dnow,detail_set)
  if locks1!=locks0 or outer1!=outer0 or between1!=between0:raise ValueError('outer patch boundary or seam changed')
  if not identical(trial['points_world_m'][anchors],anchor_xyz):raise ValueError('typed anchor coordinates changed')
  op_record={k:v for k,v in op.items()};op_record['new_faces']=[list(map(int,t)) for t in op['new_faces']]
  op_record['old_faces']=current['source_O_faces'][rows].tolist();op_record['quality_541']=qual
  op_record['exact159_delta']={'old_by_bank':{f'{k[0]}:{k[1]}':old_by[k] for k in bank_keys if old_by[k]},'new_by_bank':{f'{k[0]}:{k[1]}':new_by[k] for k in bank_keys if new_by[k]},'reduction':int(reduction),'no_bank_increase':True}
  op_record['self_screen']=incremental.report
  accepted.append(op_record);current=trial;used_faces.update(rows);used_nodes.update(quad)
  progress('accepted_operation',accepted=len(accepted),edge=list(edge),replacement_pair_reduction=int(reduction),self_pairs_after_incremental=len(state.pairs))
  qcurrent=q.face_quality(points_mm,ref_mm,current['source_O_faces'],current['area_reference_037_source_O_faces'])
 # Rebuild an exact replacement-face sum for the entire chosen batch. Unchanged skin-face pairs reuse the pinned 022 report.
 changed_rows=sorted(used_faces)
 final_replacement=counter.count_faces(changed_rows,current['source_O_faces'][changed_rows]) if changed_rows else {k:{'count':0,'triangle_pairs':[]} for k in bank_keys}
 old_batch={k:sum(face_base_counts[k].get(r,0) for r in changed_rows) for k in bank_keys}
 new_batch={k:int(final_replacement[k]['count']) for k in bank_keys}
 final_keep,final_reason,final_reduction=sel._strict_decrease(old_batch,new_batch) if accepted else (False,'no_operations_selected',0)
 if accepted and not final_keep:raise ValueError(f'combined exact 159 replacement set failed: {final_reason}')
 if state.pairs:raise ValueError('incremental self state not empty after accepted batch')
 # Replay-filter optional self-pair witnesses against the operation manifest, preserving the current base.
 rolled_back=[]
 if excluded_faces:
  keep_ops=[]
  for op in accepted:
   if excluded_faces.intersection(map(int,op['face_rows'])):rolled_back.append(op)
   else:keep_ops.append(op)
  replay={k:np.array(v,copy=True) for k,v in base.items()}
  for op in keep_ops:
   rows=list(map(int,op['face_rows']))
   if not np.array_equal(replay['source_O_faces'][rows],np.asarray(op['old_faces'],dtype=replay['source_O_faces'].dtype)):raise ValueError('rollback replay old faces mismatch')
   replay=mod405.apply_op(replay,op)
  current=replay;accepted=keep_ops
  changed_rows=sorted({int(r) for op in accepted for r in op['face_rows']})
  final_replacement=counter.count_faces(changed_rows,current['source_O_faces'][changed_rows]) if changed_rows else {k:{'count':0,'triangle_pairs':[]} for k in bank_keys}
  old_batch={k:sum(face_base_counts[k].get(r,0) for r in changed_rows) for k in bank_keys}
  new_batch={k:int(final_replacement[k]['count']) for k in bank_keys}
  final_keep,final_reason,final_reduction=sel._strict_decrease(old_batch,new_batch) if accepted else (False,'all_operations_rolled_back',0)
  if accepted and not final_keep:raise ValueError('witness-filtered remainder no longer strictly reduces total exact pairs')
 owner402.validate_candidate(current)
 if not np.array_equal(current['source_O_faces'],current['area_reference_037_source_O_faces']):raise ValueError('final current/037 face sync failed')
 if not identical(current['points_world_m'],points_before):raise ValueError('final points not bitwise identical')
 if not identical(current['points_world_m'][anchors],anchor_xyz):raise ValueError('final typed anchors changed')
 if witness_set:=excluded_faces:
  if witness_set.intersection(changed_rows):raise ValueError('rollback left an excluded self witness face in changed set')
 if any(len(v)>2 for v in owner402.edge_map(current['source_O_faces']).values()):raise ValueError('nonmanifold edge introduced')
 changed_npz=[k for k in base if not identical(base[k],current[k])]
 allowed={
  'source_O_faces','area_reference_037_source_O_faces','face_source_ancestry_indptr','face_source_ancestry_face_rows',
  'face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows','face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows','face_fine_detail_ancestry_semantics',
  'refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows','refinement_face_parent_ancestry_semantics','refinement_face_parent_face_row','refinement_face_parent_barycentric','refinement_face_parent_barycentric_semantics',
  'parent_fine_detail_face_row','parent_fine_subface_row','parent_coarse_face_row','parent_source_face_marker'}
 if set(changed_npz)-allowed:raise ValueError(f'unexpected changed arrays: {sorted(set(changed_npz)-allowed)}')
 cand_out=outdir/'candidate.npz';np.savez_compressed(cand_out,**current)
 report={
  'status':('EXACT_185_PAIR_REDUCING_CONNECTED_FLIP_PROPOSAL_PENDING_FULL_AUDIT' if accepted else 'NO_EXACT_PAIR_REDUCING_SELF_SAFE_FLIPS'),'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,
  'input_pins':{**input_pins,**owner_pins,'exact315':EXPECTED['exact315']},'candidate_path':str(base_path),'candidate_output_path':str(cand_out),'candidate_output_sha256':sha(cand_out),
  'surface_audit_bank_count':185,'input_surface_triangle_pair_count':sum(base_bank_counts.values()),'input_crossing_face_count':len(target_rows),'input_detail_ids':detail_ids,
  'progress_path':str(progress_path),'screening':{'edge_prefilter_factor':args.prefilter_factor,'edge_validation_cap':prefilter_limit,'cheap_target_edge_count':len(cheap),'selected_for_validation':len(selected_cheap),'local_541_evaluator_parity_checked':fast_parity_checked,'flip_owner':'402/405 compatible two-face source-connected flip','quality_owner':'541 evaluate_op','exact_skin_bone_owner':'370 Exact159PatchCounter, full 185 typed banks per replacement set','self_owner':'408 incremental checker with 315 predicate','base_self_pairs_reused':0,'each_accepted_operation_checked_against_full_surface_self_AABB':True,
    'max_candidate_count':args.max_candidates,'generated_quality_eligible_count':len(proposals),'exact_pair_scored_count':min(len(proposals),args.max_candidates),'exact_pair_reducing_count':len(scored),'accepted_count':len(accepted),'self_rejections':self_rejections,'rejection_counts':dict(rejected)},
  'combined_exact159_replacement_screen':{'affected_face_rows':changed_rows,'old_pair_count':sum(old_batch.values()),'new_pair_count':sum(new_batch.values()),'strict_reduction':int(final_reduction),'status':final_reason,'per_bank':{f'{k[0]}:{k[1]}':{'old':old_batch[k],'new':new_batch[k]} for k in bank_keys if old_batch[k] or new_batch[k]}},
  'accepted_operations':accepted,'witness_filtered_out_operations':rolled_back,'excluded_self_witness_rows':sorted(excluded_faces),'exact_self_witness_rollback_cli':f"rerun with --exclude-self-pairs <audit-130/surface-self-pairs.npy> --expected-exclusion-pairs-sha256 <file-sha256> and the same input candidate/report pins",
  'fixed_coordinates_bitwise_unchanged':identical(current['points_world_m'],points_before),'typed_anchor_count':int(len(anchors)),'typed_anchor_coordinates_bitwise_unchanged':identical(current['points_world_m'][anchors],anchor_xyz),
  'current_037_faces_synchronized':bool(np.array_equal(current['source_O_faces'],current['area_reference_037_source_O_faces'])),'connected_patch_outer_and_seam_edges_preserved':True,'full_csr_ancestry_preserved_and_union_replayed':True,
  'limitations':['Proposal screen counts exact surface triangle pairs only on changed skin faces against all 185 banks; unchanged counts reuse the complete pinned base report.','Incremental self checks screen each accepted operation against the whole unchanged surface, but are not a substitute for final full 130 and 159 audits.','Fixed-coordinate flips preserve nodal containment coordinates, but no post-topology full admission is claimed.']}
 (outdir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 progress('proposal_complete',accepted=len(accepted),candidate_sha256=report['candidate_output_sha256'],changed_pair_reduction=int(final_reduction),status=report['status'])
 print(json.dumps({'status':report['status'],'candidate':str(cand_out),'candidate_sha256':report['candidate_output_sha256'],'input_pair_count':report['input_surface_triangle_pair_count'],'changed_pair_reduction':final_reduction,'accepted':len(accepted),'self_rejections':len(self_rejections),'report':str(outdir/'report.json')},indent=2))
if __name__=='__main__':main()
