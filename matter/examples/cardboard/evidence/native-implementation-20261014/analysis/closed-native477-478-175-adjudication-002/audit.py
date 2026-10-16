from pathlib import Path
import csv,hashlib,json
B=Path('/Users/home/cardboard-implementation-20261014'); O=B/'analysis/closed-native477-478-175-adjudication-002'; assert not O.exists();O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
I=B/'analysis/closed-macmini-transfer-inventories-031-002/remote-inventories.json'; inv=json.loads(I.read_text())
roots={'native477':B/'solver/macmini-warm-cgls-control0-resume-477','native478':B/'solver/macmini-warm-cgls-uninterrupted-478','native175':B/'native/sparse-private-scan-175-buildfix003-command-001'}
receipts={'native477':B/'analysis/native476477-macmini-host-command-and-prefix-001/execution-477/receipt.json','native478':B/'analysis/native478-uninterrupted-command-001/execution-478/receipt.json','native175':roots['native175']/'executions/attempt-0001/receipt.json'}
transfers={}
for k,root in roots.items():
 n=size=omitted=0
 for f in inv[k]['files']:
  if f['path'].startswith('inputs/'):
   omitted+=1;continue
  p=root/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],(k,f['path']);n+=1;size+=f['bytes']
 assert sha(receipts[k])==inv[k]['terminal_receipt_sha256']
 transfers[k]={'remote_root':inv[k]['root'],'local_root':str(root),'files_verified':n,'bytes_verified':size,'duplicate_inputs_preserved_remote':omitted,'receipt_sha256':sha(receipts[k])}
r477=json.loads(receipts['native477'].read_text());assert r477['wrapper_exit_code']==2 and not r477['timed_out'] and not r477['space_abort'] and not r477['errors']
n477=json.loads((roots['native477']/'receipt.json').read_text());assert n477['exit_code']==2 and n477['source_inputs_unchanged'] and not n477['timed_out']
rows=list(csv.DictReader((roots['native477']/'output/observations.csv').open())); assert len(rows)==40
by={int(r['step']):{} for r in rows}
for r in rows:by[int(r['step'])][int(r['environment'])]=r
assert set(by)==set(range(1,21)) and all(set(x)=={0,1} for x in by.values())
for s in range(1,20):
 for r in by[s].values():assert r['step_accepted']=='1' and r['status_code']=='0' and float(r['certificate_residual'])<=5e-7 and float(r['min_J'])>0
assert by[20][0]['step_accepted']=='0' and by[20][0]['status_code']=='10'
assert by[20][1]['step_accepted']=='1'
assert by[20][0]['min_J']==by[19][0]['min_J'] and by[20][0]['measured_right_grip_angle_deg']==by[19][0]['measured_right_grip_angle_deg']
counts={}; last=[]; late=[]
for l in (roots['native477']/'stderr.log').open():
 k,sep,v=l.partition('=')
 if sep and v.startswith('{'):
  counts[k]=counts.get(k,0)+1
  if k=='fem_true_residual_cycle':
   x=json.loads(v)
   if x['root']==20 and x['restart_cycle']==63:last.append(x)
  elif k=='fem_base_cgls_direction':
   x=json.loads(v)
   if x['root']==20 and x['newton_iteration']>=48:late.append(x)
assert len(last)==64 and len(late)==16
r478=json.loads(receipts['native478'].read_text());assert r478['wrapper_exit_code']==134 and not r478['timed_out'] and not r478['space_abort']
e478=(roots['native478']/'stderr.log').read_text(); assert 'requires_true_residual_certification_for_the_same_selected_newton_state' in e478 and 'Command encoder released without endEncoding' in e478
p175=roots['native175']/'executions/attempt-0001';r175=json.loads(receipts['native175'].read_text()); assert r175['exit_code']==0
assert sha(p175/'stdout.log')==r175['stdout_sha256'] and sha(p175/'stderr.log')==r175['stderr_sha256']
assert (p175/'stderr.log').stat().st_size==0
out175=(p175/'stdout.log').read_text(); assert '12910103' in out175
report={'schema':'numi.cardboard.closed-native477-478-175-adjudication.v1','transfers':transfers,'remote_inventory_sha256':sha(I),
 'native477':{'verdict':'FAIL_full_schedule_nonlinear_acceptance_control20','full_protocol_complete':False,'accepted_controls_in_restored_suffix':19,'control0_checkpoint_separate':True,'failed_observations':by[20], 'control19_observations':by[19], 'final_control20_newton63_fgmres_trace':last[-1], 'final_control20_cgls_candidate_trace':late[-1], 'log_record_counts':counts,'timed_out':False,'accepted_geometry_retained_after_failed_trial':True,'diagnostic_material_operator_capture_requested':False,'interpretation':'Small fresh linear candidate residual does not establish nonlinear convergence. No Armijo trial trace was requested, so the accepted update fractions and failing warm-state material branches must be captured before diagnosing their cause.'},
 'native478':{'verdict':'INVALID_CONFIGURATION_encoder_teardown_abort_before_first_control','wrapper_receipt':r478,'cause':'Removing NM_FEM_TRUE_RESIDUAL_ROOT made BASE_CAUCHY all:all inadmissible despite CERTIFY=1; common rejection returned with a live encoder. This is not a physical-solver or parity result.','stderr':e478},
 'native175':{'verdict':'PASS_native_hierarchical_sparse_scan_component_only','receipt':r175,'native_checks':12910103,'cases':49,'physical_transactions':0,'shipping_contact_or_assembly_qualified':False},
 'qualification_axes':{k:False for k in ['solver_correctness','numerical_convergence','component_calibration','analogue_validation','physical_assembly','crush_impact_prediction']},'active_native479_excluded':True}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(O/'control20-final-restart-history.json').write_text(json.dumps(last,indent=2)+'\n')
(O/'control20-late-cgls-history.json').write_text(json.dumps(late,indent=2)+'\n')
(O/'audit.py').write_bytes(Path(__file__).read_bytes())
(O/'README.md').write_text('Closed remote native runs independently checked against byte inventories. 477 fails the requested full coupon; 478 is an invalid diagnostic configuration followed by encoder teardown abort; 175 passes only a native scan component differential checker. All six qualification axes remain open. Active479 excluded.\n')
(O/'manifest.json').write_text(json.dumps({'files_sha256':{p.name:sha(p) for p in sorted(O.iterdir())}},indent=2)+'\n')
print(json.dumps({'package':str(O),'manifest_sha256':sha(O/'manifest.json'),'transfers':transfers,'native477_final_residual':by[20][0]['certificate_residual'],'native175_checks':12910103},indent=2))
