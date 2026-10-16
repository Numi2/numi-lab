from pathlib import Path
import datetime,hashlib,json,csv
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=Q/'analysis/native531532-complete-replay-parity-002'
A=Q/'solver/control17-root17-newton34-replay-531';B=Q/'solver/control17-root17-diagnostic-off-532';F=Q/'solver/fresh-damped-cgls-complete-coupon-521'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def traces(case,key):
 rows=[]
 for line in (case/'stderr.log').read_text().splitlines():
  if line.startswith(key+'='):
   r=json.loads(line.split('=',1)[1])
   if r.get('root')==17:rows.append(r)
 return rows
def main():
 assert not (P/'report.json').exists()
 receipts={}
 for c in [A,B,F]:
  r=json.loads((c/'receipt.json').read_text());assert r['execution_started'] and r['source_inputs_unchanged'] and r['exit_code']==2 and not r['timed_out'] and not r['termination_errors']
  receipts[c.name]={'sha256':sha(c/'receipt.json'),'exit_code':r['exit_code'],'wall_seconds':r['wall_seconds']}
 pA=json.loads((A/'prediction.json').read_text());pB=json.loads((B/'prediction.json').read_text())
 assert pA['source_snapshot']==pB['source_snapshot'] and pA['runtime_dir']==pB['runtime_dir']
 aenv=dict(pA['environment']);benv=dict(pB['environment'])
 for n in ['NM_FEM_OPERATOR_FD_SELECTOR','NM_FEM_OPERATOR_FD_CAPTURE_PATH','NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT']:assert n in aenv;aenv.pop(n)
 assert aenv==benv and aenv['NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR']=='all:3'
 invA={str(p.relative_to(A/'output')):sha(p) for p in (A/'output').rglob('*') if p.is_file()}
 invB={str(p.relative_to(B/'output')):sha(p) for p in (B/'output').rglob('*') if p.is_file()}
 assert invA.keys()==invB.keys()
 compared={k:h for k,h in invA.items() if k!='coupon-resume-segment.json'}
 assert compared=={k:h for k,h in invB.items() if k!='coupon-resume-segment.json'}
 for c in [A,B]:
  r=json.loads((c/'output/coupon-resume-segment.json').read_text())
  assert r['accepted_state_restored_from_native_snapshot'] and not r['accepted_prefix_reconstructed_from_obj_or_json'] and r['checkpoint_sha256']=='fbee60ec02c89cd877afc358cd575371927b04e1a46a42be0ee9b61793aeba51'
  assert r['source_prefix_accepted_control_count']==17 and r['segment_accepted_controls']==0 and r['segment_start_control_index']==17 and not r['trajectory_parity_established'] and not r['full_protocol_complete'] and not r['physical_validation']
 fc={k:h for k,h in compared.items() if k not in ['result.json','observations.csv']}
 for k,h in fc.items():assert sha(F/'output'/k)==h,k
 af=list(csv.DictReader((A/'output/observations.csv').open()));ff=list(csv.DictReader((F/'output/observations.csv').open()))
 assert len(af)==2 and af==[r for r in ff if r['step']=='17']
 assert af[0]['status_code']=='10' and af[0]['failing_index']=='34' and af[0]['step_accepted']=='0' and af[1]['step_accepted']=='1'
 counts={}
 for key in ['fem_true_residual_cycle','fem_base_cgls_direction','fem_base_cgls_enabled','fem_base_cgls_damped_enabled']:
  ta,tb,tf=[traces(c,key) for c in [A,B,F]]
  assert ta==tb==tf,key
  counts[key]=len(ta)
 assert counts['fem_true_residual_cycle']>0
 capture=Path(pA['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH'])
 with capture.open() as f:h=json.loads(next(f))
 assert h['production_inputs_bitwise_unchanged'] and h['root']==17 and h['newton_iteration']==34
 report={'schema':'numi.cardboard.native531532-replay-parity.v1','verdict':'PASS','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipts':receipts,'capture_sha256':sha(capture),'capture_production_inputs_bitwise_unchanged':True,'observer_on_off_exact_production_files':len(compared),'observer_on_off_files_sha256':compared,'same_files_as_original521':len(fc),'same_original521_observation_rows':2,'all_control17_common_trace_records_equal':counts,'native_checkpoint_replay_parity_at_failed_control17':True,'accepted_prefix_0_to16_separately_bound_by':'analysis/native530-native521-accepted-prefix-parity-001','full_protocol_complete':False,'six_qualification_axes':{k:False for k in ['solver_correctness','numerical_convergence','component_calibration','analogue_validation','physical_assembly','crush_impact_prediction']},'scope':'Exact observer parity for the complete failed resumed control17 segment and original521, including rejected geometry/material/conditioning and full common solver traces. Accepted prefix0..16 qualification is separately measured; no completed loading/release/relaxation claim.'}
 (P/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 (P/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.replay-parity-files.v1','files_sha256':{p.name:sha(p) for p in sorted(P.iterdir()) if p.is_file()}},indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='observer_on_off_files_sha256'}))
if __name__=='__main__':main()
