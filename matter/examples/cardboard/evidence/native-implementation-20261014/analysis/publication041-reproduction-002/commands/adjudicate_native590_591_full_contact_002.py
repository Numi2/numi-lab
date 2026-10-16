from pathlib import Path
import json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native590-591-full-contact-root-adjudication-002';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cases=[q/'native/activation-legacy-control-off-590',q/'native/activation-full-residual-control-on-591']
bindings={}; reports=[]
for c in cases:
 r=json.loads((c/'receipt.json').read_text());assert r['execution_started'] and r['source_inputs_unchanged'] and r['exit_code']==1 and not r['timed_out']
 o=c/'output/dense';attempts=[json.loads(p.read_text()) for p in sorted(o.glob('transaction-*.json'))]
 assert [a['requested_control_step'] for a in attempts]==[1199,1200,1201,1202]
 assert [a['accepted'] for a in attempts]==[True,True,True,False]
 a=attempts[-1]
 assert a['before']['control_step']==a['after']['control_step']==1201
 assert a['before']['accepted_state_vector_sha256']==a['after']['accepted_state_vector_sha256']
 assert a['before']['vector_counts']==a['after']['vector_counts']
 assert a['after']['statuses'][0]['code']==10 and a['after']['statuses'][0]['failing_index']==7
 trace=[]
 for line in (c/'stderr.log').read_text().splitlines():
  if line.startswith('fem_residual_armijo_iteration='):
   j=json.loads(line.split('=',1)[1])
   if j.get('root')==1202:trace.append(j)
 assert len(trace)==16 and trace[7]['newton_iteration']==7
 assert trace[7]['environments'][0]['trial_status']['code']==6
 reports.append({'case':str(c.relative_to(q)),'exit_code':r['exit_code'],'accepted_controls':[1199,1200,1201],'failed_control':1202,'failed_newton':7,'accepted_state_vector_hashes_preserved_on_failure':True,'accepted_cursor_preserved_on_failure':True,'failed_status':a['after']['statuses'][0],'failed_certificate':a['after']['solver_certificates'][0],'trace_records':trace})
 for p in [c/'receipt.json',c/'stderr.log',*sorted(o.glob('transaction-*'))]:bindings[str(p.relative_to(q))]={'sha256':sha(p),'bytes':p.stat().st_size}
side=cases[1]/'output/dense/activation-transaction-000000-scene-commit.json';s=json.loads(side.read_text())
assert s['matter_accepted'] and s['full_residual_mode_selected'] and not s['full_residual_mode_skipped'] and s['metalworld_diagnostics_succeeded']
assert s['accepted_steps_before']==0 and s['accepted_steps_after']==1 and s['published_scene_matches_host_commit']
for k in ['input_scene','kinematic_targets','metalworld_published_scene','runner_scene_after','final_q','final_v']:
 filekey=k+'_file'
 if filekey in s:
  p=side.parent/s[filekey];assert sha(p)==s[k+'_sha256'];bindings[str(p.relative_to(q))]={'sha256':sha(p),'bytes':p.stat().st_size}
bindings[str(side.relative_to(q))]={'sha256':sha(side),'bytes':side.stat().st_size}
common=[]
for p in sorted((cases[0]/'output/dense').glob('transaction-*')):
 other=cases[1]/'output/dense'/p.name
 assert other.exists() and sha(other)==sha(p)
 common.append(p.name)
for p in [q/'snapshots/source-activation-full-residual-armijo-244-002/frozen-source.json',q/'build-source244-activation-002/source-build.json',Path(__file__)]:
 bindings[str(p.relative_to(q))]={'sha256':sha(p),'bytes':p.stat().st_size}
report={'schema':'numi.cardboard.native590-591-root-adjudication.v1','verdict':'ACTIVATION_ACCEPTED_SUBSEQUENT_FULL_CONTACT_HISTORY_FAILED','cases':reports,'activation_host_scene_observation':s,'common_transaction_files_identical':common,'classification':'Native coupled rigid/FEM/cohesive fixture; no shipping box qualification','limits':['1199 activation is accepted with full residual mode selected and published host scene committed.','1200 and1201 accepted;1202 Newton7 has no accepted private trial and fails.','Accepted Matter vector hashes and accepted cursor remain unchanged after failure; diagnostic status and certificates change as expected.','Later freebody host scene and raw transient Metal body buffer are not covered by activation-only1199 sidecar; extended fixture observer requested separately.','Private-stage cause is not identified by null norm alone. Earlier per-feature barrier Armijo fixture does not establish this cause; full-contact policy selects feasibility-only contact vetting.','Legacy-off and on fixtures produce identical Matter transaction outputs on these actions; this is not a general policy-equivalence proof.'],'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(d/'manifest.json').write_text(json.dumps({'inputs':bindings,'report_sha256':sha(d/'report.json')},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'common_transaction_files':len(common),'activation_accepted':True,'subsequent_control1202_failed':True}))
