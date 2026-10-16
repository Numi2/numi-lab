from pathlib import Path
import json,hashlib,csv,struct,math,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native596-598-root-adjudication-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=[];reports={}
c=q/'solver/strict-state-source240-control19-final-newton-diagnostic-596';base=q/'solver/strict-state-source240-newton128-full-coupon-585'
r=json.loads((c/'receipt.json').read_text());assert r['exit_code']==2 and r['source_inputs_unchanged'] and not r['timed_out']
rows=lambda p:list(csv.DictReader(p.open()))
a=[x for x in rows(base/'output/observations.csv') if int(x['step'])>=16];b=rows(c/'output/observations.csv');assert a==b and len(b)==8
names=[p.name for p in (c/'output').iterdir() if p.is_file() and (p.name.startswith(('accepted_step_','rejected_step_','material_state_')) or p.name in ['compiled.nmatterpack','mesh.json','manifest.json'])]
same=[]
for n in sorted(names):assert sha(c/'output'/n)==sha(base/'output'/n),n;same.append(n);inputs.extend([c/'output'/n,base/'output'/n])
bins=[]
for p in sorted((c/'output/rejected-conditioning-000020').glob('*.bin')):
 x=base/'output/rejected-conditioning-000020'/p.name;assert sha(p)==sha(x);bins.append(p.name);inputs.extend([p,x])
capture=c/'diagnostics/capture-operator-fd-capture/capture';assert sha(capture)==r['diagnostic_capture_validation']['files'][0]['sha256']
reports['596']={'verdict':'PASS_DIAGNOSTIC_PRODUCTION_SUFFIX_PARITY_TARGET_STILL_FAILS','controls':[16,17,18,19],'observation_rows_exact':8,'production_files_exact':same,'rejected_conditioning_bins_exact':bins,'capture_sha256':sha(capture),'capture_bytes':capture.stat().st_size,'scope':'Actual strict-policy control19 Newton127. Raw cached matrices are captured, but this old observer action is uncached; cached native action parity is not measured.','target_failed':True}
inputs.extend([c/'receipt.json',base/'receipt.json',capture,c/'output/observations.csv',base/'output/observations.csv'])
c=q/'native/freebody-newton7-private-trial-summary-597';base=q/'native/activation-full-residual-control-on-591';r=json.loads((c/'receipt.json').read_text());assert r['exit_code']==1 and r['source_inputs_unchanged'] and not r['timed_out']
same=[]
for p in sorted((base/'output/dense').glob('transaction-*')):
 x=c/'output/dense'/p.name;assert sha(p)==sha(x),p.name;same.append(p.name);inputs.extend([p,x])
jp=c/'diagnostics/capture-coupled-trial-summary/capture.summary.json';bp=jp.with_suffix('.bin');j=json.loads(jp.read_text());raw=bp.read_bytes();assert len(raw)==j['payload']['byte_count'] and sha(bp)==j['payload']['sha256']
arrays={a['name']:a for a in j['arrays']}
def payload(name):
 a=arrays[name];v=raw[a['offset']:a['offset']+a['byte_count']];assert hashlib.sha256(v).hexdigest()==a['sha256'];return v
for name in arrays:payload(name)
v=payload('armijo_trial_status_checkpoints');trials=[]
for i in range(8):
 states=[struct.unpack_from('<8I4f',v,(i*8+s)*48) for s in range(8)]
 first=next(s for s,x in enumerate(states) if x[0])
 assert first==5 and states[first][0:4]==(6,0,0,78) and states[first][8:]==(16.,99.,1.,0.)
 m=struct.unpack('<4f',payload('trial_'+str(i)+'_metrics_predecision'))
 assert abs(m[2]-2.**(-i))<1e-12 and m[3]==0
 trials.append({'trial':i,'tested_alpha':m[2],'first_failed_stage':j['status_stage_names'][first],'status_code':6,'failed_index':78,'diagnostics':list(states[first][8:]),'trial_residual_is_nonfinite':not math.isfinite(m[1])})
reports['597']={'verdict':'PASS_OBSERVER_PARITY_FIRST_FAILURE_LOCALIZED','production_transaction_files_exact':same,'accepted_controls':[1199,1200,1201],'failed_control':1202,'failed_newton':7,'all_array_hashes_verified':len(arrays),'trials':trials,'interpretation':'All8 trials pass geometry, bulk, cohesive, rigid contact, and private segment deformable_sweep_ccd. First failure is deformable_self_narrowphase for parent primitives16/99, valid mortar partitions but uncertified findMortarComplementContact. Individual feature cause remains open; no tolerance/contact bypass is justified.','physical_history_completed':False}
inputs.extend([c/'receipt.json',base/'receipt.json',jp,bp,c/'stderr.log'])
c=q/'native/production-contact-compactor-fixture-fixed-598';r=json.loads((c/'receipt.json').read_text());out=(c/'stdout.log').read_text();assert r['exit_code']==0 and r['source_inputs_unchanged'] and not r['timed_out'];assert 'FEM/rigid/history_bitwise_parity=PASS' in out and 'overflow_prefix_rejected=PASS' in out and 'physical_transactions=0' in out
reports['598']={'verdict':'PASS_PRODUCTION_CONTACT_KERNEL_COMPONENT_PARITY_AND_OVERFLOW','stdout':out.strip(),'physical_transactions':0,'scope':'Manufactured E2/P7/4active-per-env production contact-sample compaction. Corrected fixture zero-seeds both paths; overflow sentinels unchanged. No full-runtime/shipping memory qualification.'}
inputs.extend([c/'receipt.json',c/'stdout.log',q/'snapshots/source-contact-compaction-fixture-249-001/frozen-source.json',q/'analysis/source249-contact-compaction-fixture-001/build-001/source-build.json',Path(__file__)])
report={'schema':'numi.cardboard.native596-598-root-adjudication.v1','results':reports,'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs},'report_sha256':sha(d/'report.json')},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'596files':len(reports['596']['production_files_exact']),'596bins':len(reports['596']['rejected_conditioning_bins_exact']),'597firstfailure':'deformable_self_narrowphase','598pass':True}))
