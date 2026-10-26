#!/usr/bin/env python3
"""Run bounded exact 159/self-safe topology batches with compositional receipts.
Each changed skin face is recounted against every typed bank; unchanged face-pair
identities inherit from a pinned complete 159 report. 130 zero-self state carries
forward only through accepted 408 incremental checks. No full final audit is run.
"""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
D=ROOT/'D'; OWNER370=D/'connected-remesh-reuse-370/select_bone_reducing_flips_370.py'
WRAPPER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/topology/topology-bone-aware-025.py')
WRAPPER_SHA='9123965621161beb6408f6c9663d8d36968cf1113c3a97106e968ea1c754d84e'
PRED_SHA='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def load_npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def identical(a,b):
 x,y=np.ascontiguousarray(a),np.ascontiguousarray(b);return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
def pair_total(report):return sum(int((b.get('result') or {}).get('count',0)) for b in report['banks'])
def atomic_json(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)
def compose159(previous,prev_hash,next_candidate,next_hash,proposal_report,sel,banks,bank_keys,counter):
 before=load_npz(previous);after=load_npz(next_candidate)
 if not identical(before['points_world_m'],after['points_world_m']):raise ValueError('coordinate bytes changed in topology-only chain')
 if not np.array_equal(after['source_O_faces'],after['area_reference_037_source_O_faces']):raise ValueError('current/037 topology desynchronized')
 old_faces=np.asarray(before['source_O_faces']);new_faces=np.asarray(after['source_O_faces'])
 changed=np.flatnonzero(np.any(old_faces!=new_faces,axis=1)).astype(np.int64)
 if proposal_report.get('input_pins',{}).get('candidate')!=prev_hash or proposal_report.get('candidate_output_sha256')!=next_hash:raise ValueError('proposal report is not bound to the supplied input/output candidates')
 accepted=proposal_report.get('accepted_operations',[])
 manifest_rows=set()
 for op in accepted:
  rows=list(map(int,op['face_rows']));manifest_rows.update(rows)
  if not np.array_equal(old_faces[rows],np.asarray(op['old_faces'],dtype=old_faces.dtype)):raise ValueError('proposal old-face receipt mismatch')
  if not np.array_equal(new_faces[rows],np.asarray(op['new_faces'],dtype=new_faces.dtype)):raise ValueError('proposal new-face receipt mismatch')
  screen=op.get('self_screen',{})
  if screen.get('status')!='EXACT_INCREMENTAL_GLOBAL_SELF_AUDIT' or screen.get('candidate_sha256')!=prev_hash or screen.get('surface_predicate_sha256')!=PRED_SHA or screen.get('exact_PLC_surface_pairs')!=0 or screen.get('introduced_pairs') or screen.get('no_new_pairs') is not True:raise ValueError('accepted operation lacks a passing current-candidate 408 incremental self receipt')
 if set(map(int,changed))!=manifest_rows:raise ValueError('candidate changed-face set differs from accepted operation manifest')
 previous_report=json.loads(Path(previous_report_path).read_text())
 if previous_report.get('candidate_sha256')!=prev_hash:raise ValueError('previous exact pair inventory is not bound to previous candidate')
 base_total=pair_total(previous_report)
 current_counts=counter.count_faces(changed,new_faces[changed]) if len(changed) else {k:{'count':0,'triangle_pairs':[]} for k in bank_keys}
 new_report=json.loads(json.dumps(previous_report))
 by_key={(int(b['body']),int(b['stable_bone_id'])):b for b in new_report['banks']}
 if set(by_key)!=set(bank_keys):raise ValueError('typed bank inventory differs from pinned 185-bank base')
 for key in bank_keys:
  row=by_key[key];result=row.get('result')
  prior=[] if result is None else [tuple(map(int,p)) for p in result['triangle_pairs']]
  kept=[p for p in prior if p[0] not in set(map(int,changed))]
  refreshed=[tuple(map(int,p)) for p in current_counts[key]['triangle_pairs']]
  merged=sorted(kept+refreshed)
  if len(merged)!=len(set(merged)):raise ValueError(f'duplicate exact pair while composing bank {key}')
  row.pop('skin_aabb_active',None);row.pop('bank_aabb_active',None)
  row['result']={'count':len(merged),'triangle_pairs':[list(p) for p in merged],'allowed_shared_vertex_or_edge_pairs':0}
 new_report['candidate_sha256']=next_hash
 new_report['complete']=True;new_report['geometry_or_native_admission']=False
 new_report['composed_exact_pair_inventory']={
  'full_159_execution_for_composed_candidate':False,
  'method':'pinned complete previous 185-bank pair identities with exact Exact159PatchCounter recomputation for every changed skin face',
  'previous_candidate_sha256':prev_hash,'previous_report_sha256':sha(previous_report_path),
  'changed_skin_face_rows':list(map(int,changed)),'exact_counter_owner_sha256':sha(OWNER370),
  'all_185_banks_recounted_for_changed_rows':True,
 }
 # Independent owner parser verifies every pair identity, typed bank, and count after composition.
 sel._pair_rows_and_counts(new_report,candidate_sha256=next_hash,face_count=len(new_faces),banks=banks,expected_bank_keys=bank_keys)
 delta=base_total-pair_total(new_report)
 declared=proposal_report.get('combined_exact159_replacement_screen',{})
 if accepted and delta<=0:raise ValueError('composed exact 159 inventory does not confirm strict reduction')
 if accepted and int(declared.get('strict_reduction',-1))!=delta:raise ValueError('composed full report delta differs from proposal replacement-face screen')
 return new_report,changed,base_total,pair_total(new_report)

def main():
 global previous_report_path
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('candidate','audit159','audit130','self-pairs130','output-root'):
  ap.add_argument('--'+name,required=True,type=Path)
 for name in ('candidate','audit159','audit130','self-pairs130'):
  ap.add_argument('--expected-'+name.replace('_','-')+'-sha256',required=True)
 ap.add_argument('--rounds',type=int,default=4);ap.add_argument('--max-candidates',type=int,default=128);ap.add_argument('--max-accept',type=int,default=64);ap.add_argument('--prefilter-factor',type=int,default=3);ap.add_argument('--progress-every',type=int,default=8)
 ap.add_argument('--seed-base-candidate',type=Path);ap.add_argument('--seed-proposal-report',type=Path,help='compose an existing exact-screened 024 proposal on a base with a full exact report before starting new rounds')
 a=ap.parse_args();outroot=a.output_root.resolve()
 if outroot.exists():raise FileExistsError(outroot)
 outroot.mkdir(parents=True)
 for name in ('candidate','audit159','audit130','self_pairs130'):
  path=getattr(a,name);actual=sha(path);expected=getattr(a,'expected_'+name+'_sha256')
  if actual!=expected:raise ValueError(f'{name} input SHA mismatch: {actual}')
 if sha(WRAPPER)!=WRAPPER_SHA:raise ValueError('025 wrapper source pin mismatch')
 candidate=a.candidate.resolve();previous_report_path=a.audit159.resolve();audit130=a.audit130.resolve();self_pairs=a.self_pairs130.resolve()
 initial_candidate_sha=sha(candidate);initial_audit130_sha=sha(audit130)
 state=load_npz(candidate);anchors=np.asarray(state['typed_coarse_anchor_output_rows_018'],dtype=np.int64);anchor_xyz=state['points_world_m'][anchors].copy()
 if len(anchors)!=237 or len(np.unique(anchors))!=237:raise ValueError('typed anchor inventory mismatch')
 # Load the frozen all-185 exact pair counter once; coordinates and bank geometry remain invariant throughout the chain.
 sel=loadmod('select_bone_reducing_flips_370_chain025',OWNER370)
 flip,surface_owner,bank_helper,exact159,runtime=sel.load_pinned_runtime()
 raw=bank_helper.read_bank_inputs(sel.BANK_DIR_DEFAULT,sel.EVIDENCE_ROOT_DEFAULT)
 banks,bank_keys=sel.typed_banks_from_159_inputs(raw)
 counter=sel.Exact159PatchCounter(state['points_world_m'],banks,exact159,surface_owner.records)
 if np.load(self_pairs,allow_pickle=False).shape!=(0,2):raise ValueError('initial self-pair witness file is not empty')
 # Optional seed: compose an already screened fixed-coordinate batch on top of a genuinely full-audited base.
 if a.seed_proposal_report:
  if not a.seed_base_candidate:raise ValueError('--seed-proposal-report requires --seed-base-candidate')
  seed_base=a.seed_base_candidate.resolve();seed_report_path=a.seed_proposal_report.resolve();full130=json.loads(audit130.read_text())
  base_sha=sha(seed_base);seed_sha=sha(candidate);seed_proposal=json.loads(seed_report_path.read_text())
  if seed_proposal.get('input_pins',{}).get('candidate')!=base_sha or seed_proposal.get('candidate_output_sha256')!=seed_sha:raise ValueError('seed proposal is not bound to the supplied base/output pair')
  if full130.get('complete_surface_audit') is not True or full130.get('candidate_sha256')!=base_sha or int(full130.get('exact_PLC_surface_pairs',-1))!=0 or full130.get('surface_predicate_sha256')!=PRED_SHA:raise ValueError('seed base lacks complete full zero-self 130 evidence')
  previous_report_path=a.audit159.resolve()
  composed,changed,before_count,after_count=compose159(seed_base,base_sha,candidate,seed_sha,seed_proposal,sel,banks,bank_keys,counter)
  seed_dir=outroot/'round-000-seed';seed_dir.mkdir()
  composed_path=seed_dir/'composed-exact-159-inventory.json';atomic_json(composed_path,composed)
  op_checks=[];introduced=0
  for op in seed_proposal.get('accepted_operations',[]):
   screen=op.get('self_screen',{});n=len(screen.get('introduced_pairs',[]));introduced+=n
   if screen.get('status')!='EXACT_INCREMENTAL_GLOBAL_SELF_AUDIT' or screen.get('candidate_sha256')!=base_sha or screen.get('surface_predicate_sha256')!=PRED_SHA or screen.get('exact_PLC_surface_pairs')!=0 or screen.get('no_new_pairs') is not True:raise ValueError('seed operation lacks a passing 408 incremental self receipt')
   op_checks.append({'edge':op.get('edge'),'face_rows':op.get('face_rows'),'introduced_self_pair_count':n})
  if introduced:raise ValueError('seed proposal introduced self pairs')
  initial_audit130_sha=sha(audit130)
  seed_cert={'type':'INCREMENTAL_ZERO_SELF_CARRYFORWARD','candidate_sha256':seed_sha,'zero_self_pair_count':0,'surface_predicate_sha256':PRED_SHA,'base_full_130_audit_sha256':initial_audit130_sha,'base_full_130_candidate_sha256':base_sha,'incremental_chain':[{'input_candidate_sha256':base_sha,'output_candidate_sha256':seed_sha,'proposal_report_sha256':sha(seed_report_path),'accepted_operation_count':len(op_checks),'operation_checks':op_checks,'introduced_self_pair_count':introduced,'final_self_pair_count':0,'full_130_execution':False}],'claim_scope':'Exact 315/408 incremental self checks from a complete zero-self base audit; not a fresh full 130 execution on the seed candidate.'}
  cert_path=seed_dir/'incremental-self-certificate.json';atomic_json(cert_path,seed_cert)
  atomic_json(seed_dir/'state.json',{'candidate':str(candidate),'candidate_sha256':seed_sha,'exact_159_inventory':str(composed_path),'exact_159_inventory_sha256':sha(composed_path),'self_state_certificate':str(cert_path),'self_state_certificate_sha256':sha(cert_path),'base_full_130_audit_sha256':initial_audit130_sha,'full_130_run_on_this_candidate':False,'full_159_run_on_this_candidate':False,'surface_pair_count_composed':after_count,'changed_face_count':int(len(changed)),'accepted_operation_count':len(op_checks),'native_admission':False,'accepted_physical_steps':0})
  previous_report_path=composed_path;audit130=cert_path;current_self_cert=seed_cert
  print(json.dumps({'stage':'seed_proposal_composed','candidate_sha256':seed_sha,'before_pairs':before_count,'after_pairs':after_count,'reduction':before_count-after_count,'changed_faces':int(len(changed)),'accepted':len(op_checks),'full_audits_on_seed':False}),flush=True)
 else:
  initial_summary=json.loads(a.audit130.read_text())
  if initial_summary.get('complete_surface_audit') is not True or initial_summary.get('candidate_sha256')!=initial_candidate_sha or int(initial_summary.get('exact_PLC_surface_pairs',-1))!=0 or initial_summary.get('surface_predicate_sha256')!=PRED_SHA:raise ValueError('chain must start from a complete exact zero-self 130 audit')
  current_self_cert=None
 chain=[];results=[]
 for index in range(1,a.rounds+1):
  round_dir=outroot/f'round-{index:03d}'
  args=[sys.executable,str(WRAPPER),'--candidate',str(candidate),'--audit159',str(previous_report_path),'--audit130',str(audit130),'--self-pairs130',str(self_pairs),
   '--expected-candidate-sha256',sha(candidate),'--expected-audit159-sha256',sha(previous_report_path),'--expected-audit130-sha256',sha(audit130),'--expected-self-pairs-sha256',sha(self_pairs),
   '--output-dir',str(round_dir),'--max-candidates',str(a.max_candidates),'--max-accept',str(a.max_accept),'--prefilter-factor',str(a.prefilter_factor),'--progress-every',str(a.progress_every)]
  atomic_json(round_dir.with_suffix('.launch.json'),{'command':args,'wrapper_sha256':WRAPPER_SHA,'input_candidate_sha256':sha(candidate),'input_audit159_sha256':sha(previous_report_path),'input_self_state_sha256':sha(audit130),'output_dir':str(round_dir),'geometry_or_native_admission':False})
  print(json.dumps({'stage':'chain_round_started','round':index,'candidate_sha256':sha(candidate),'input_pair_count':pair_total(json.loads(previous_report_path.read_text()))}),flush=True)
  proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  with (round_dir.with_suffix('.execution.log')).open('w') as log:
   assert proc.stdout is not None
   for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
  rc=proc.wait()
  if rc:raise RuntimeError(f'025 batch round {index} exited {rc}; logs preserved')
  proposal=json.loads((round_dir/'report.json').read_text());next_candidate=round_dir/'candidate.npz';next_hash=sha(next_candidate)
  if next_hash!=proposal.get('candidate_output_sha256'):raise ValueError('batch output candidate hash mismatch')
  if not np.array_equal(state['points_world_m'][anchors],anchor_xyz):raise ValueError('initial anchor snapshot changed unexpectedly')
  new_report,changed,before_pairs,after_pairs=compose159(candidate,sha(candidate),next_candidate,next_hash,proposal,sel,banks,bank_keys,counter)
  composed_path=round_dir/'composed-exact-159-inventory.json';atomic_json(composed_path,new_report)
  op_checks=[];introduced=0
  for op in proposal.get('accepted_operations',[]):
   screen=op.get('self_screen',{});pairs=screen.get('introduced_pairs',[]);introduced+=len(pairs)
   if screen.get('status')!='EXACT_INCREMENTAL_GLOBAL_SELF_AUDIT' or screen.get('candidate_sha256')!=sha(candidate) or screen.get('surface_predicate_sha256')!=PRED_SHA or screen.get('exact_PLC_surface_pairs')!=0 or screen.get('no_new_pairs') is not True:raise ValueError('batch operation lacks a passing 408 incremental self receipt')
   op_checks.append({'edge':op.get('edge'),'face_rows':op.get('face_rows'),'introduced_self_pair_count':len(pairs)})
  if introduced:raise ValueError('selected batch has introduced self pairs')
  prior_chain=[] if current_self_cert is None else current_self_cert.get('incremental_chain',[])
  base_full_audit_sha=initial_audit130_sha if current_self_cert is None else current_self_cert['base_full_130_audit_sha256']
  base_full_candidate_sha=initial_candidate_sha if current_self_cert is None else current_self_cert['base_full_130_candidate_sha256']
  cert={
   'type':'INCREMENTAL_ZERO_SELF_CARRYFORWARD','candidate_sha256':next_hash,'zero_self_pair_count':0,'surface_predicate_sha256':PRED_SHA,
   'base_full_130_audit_sha256':base_full_audit_sha,'base_full_130_candidate_sha256':base_full_candidate_sha,
   'incremental_chain':prior_chain+[{'input_candidate_sha256':sha(candidate),'output_candidate_sha256':next_hash,'proposal_report_sha256':sha(round_dir/'report.json'),'accepted_operation_count':len(op_checks),'operation_checks':op_checks,'introduced_self_pair_count':introduced,'final_self_pair_count':0,'full_130_execution':False}],
   'claim_scope':'Exact 315/408 incremental self checks from a complete initial zero-self audit; not a fresh full 130 execution on this candidate.'}
  cert_path=round_dir/'incremental-self-certificate.json';atomic_json(cert_path,cert)
  round_state={'candidate':str(next_candidate),'candidate_sha256':next_hash,'exact_159_inventory':str(composed_path),'exact_159_inventory_sha256':sha(composed_path),'self_state_certificate':str(cert_path),'self_state_certificate_sha256':sha(cert_path),'full_130_run_on_this_candidate':False,'full_159_run_on_this_candidate':False,'surface_pair_count_composed':after_pairs,'changed_face_count':int(len(changed)),'accepted_operation_count':len(op_checks),'native_admission':False,'accepted_physical_steps':0}
  atomic_json(round_dir/'state.json',round_state)
  row={'round':index,'candidate_sha256':next_hash,'before_pairs':before_pairs,'after_pairs':after_pairs,'reduction':before_pairs-after_pairs,'changed_face_count':int(len(changed)),'accepted_count':len(op_checks),'introduced_self_pair_count':introduced,'report_path':str(round_dir/'report.json'),'state_path':str(round_dir/'state.json')};results.append(row)
  print(json.dumps({'stage':'chain_round_complete',**row},sort_keys=True),flush=True)
  if not op_checks or after_pairs>=before_pairs:break
  candidate=next_candidate;previous_report_path=composed_path;audit130=cert_path;current_self_cert=cert
  state=load_npz(candidate)
 summary={'status':'BOUNDED_CHAIN_COMPLETE_PENDING_ONE_FINAL_FULL_EXACT_AUDIT','rounds_requested':a.rounds,'rounds_completed':len(results),'input_candidate_sha256':initial_candidate_sha,'input_full_130_audit_sha256':(current_self_cert['base_full_130_audit_sha256'] if current_self_cert else initial_audit130_sha),'results':results,'final_state':round_state if results else None,'method_limits':['159 counts are exact per changed face and composed with unchanged identities from a pinned complete report.','130 self-free state is incremental carry-forward, not a fresh full audit on intermediate candidates.','One full final 130 and 159 audit is required after the chain.','All coordinates and typed anchors are required bitwise unchanged across each batch.'],'native_admission':False,'accepted_physical_steps':0}
 atomic_json(outroot/'chain-summary.json',summary);print(json.dumps({'stage':'chain_complete','summary':str(outroot/'chain-summary.json'),'rounds':len(results),'final_pairs':results[-1]['after_pairs'] if results else None}),flush=True)
if __name__=='__main__':main()
