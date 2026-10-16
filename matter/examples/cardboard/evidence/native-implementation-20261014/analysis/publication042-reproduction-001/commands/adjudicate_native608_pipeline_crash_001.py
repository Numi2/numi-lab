from pathlib import Path
import json,hashlib,datetime,shutil
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-best-of-cgls-255-001';c=q/'solver/best-of255-enabled-seven-phase-608';d=q/'analysis/native608-pipeline-crash-root-adjudication-001';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
osreport=Path('/Users/n/Library/Logs/DiagnosticReports/numi-matter-cardboard-probe-2026-10-16-172329.ips');copy=d/'native608-crash.ips';shutil.copy2(osreport,copy);assert sha(osreport)==sha(copy)
header,body=copy.read_text().split('\n',1);j=json.loads(body);h=json.loads(header)
r=json.loads((c/'receipt.json').read_text());assert r['exit_code']==-11 and not r['timed_out'] and r['source_inputs_unchanged'];assert j['pid']==15101 and j['usedImages'][0]['uuid']=='a2271235-e548-38b1-b360-8a6398ffc444'
src=(s/'matter/src/runtime.mm').read_text();lines=src.splitlines()
names=['nm_fgmres_base_cgls_best_of_'+x for x in ['save','save_direction','resolve','restore_direction','measure_committed']]
occ={n:[i+1 for i,l in enumerate(lines) if '"'+n+'"' in l] for n in names}
assert all(len(v)==1 and v[0]>21000 for v in occ.values())
assert 'nm_fgmres_prepare_base_cgls_layout' in '\n'.join(lines[4600:4800])
accepted=list((c/'output').glob('accepted_step_*.obj'));assert len(accepted)==0
report={'schema':'numi.cardboard.native608-pipeline-crash-root-adjudication.v1','verdict':'FAIL_ENABLED_RUNTIME_SIGSEGV_BEFORE_FIRST_ACCEPTED_PHYSICAL_STEP','native_exit_code':-11,'source_inputs_unchanged':True,'timeout':False,'accepted_geometry_count':len(accepted),'crash_report_sha256':sha(copy),'crash_pid':j['pid'],'crash_image_uuid':j['usedImages'][0]['uuid'],'faulting_address':j['exception']['subtype'],'triggered_stack':j['threads'][j['faultingThread']]['frames'],'missing_runtime_registration_evidence':occ,'localization':'All five newly dispatched best-of kernels are absent from the runtime pipeline registration list. Native stack enters AGX setComputePipelineState through Runtime dispatchThreads. The direct-library component fixture does not test Runtime registration. Fresh source263 repair and runtime regression are required.','CPU_and_component_pass_do_not_qualify_enabled_runtime':True,'no_full_coupon_launched':True,'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
inputs=[Path(__file__),copy,c/'receipt.json',c/'stderr.log',c/'stdout.log',q/'analysis/native608-command-001/command.json',s/'frozen-source.json',s/'matter/src/runtime.mm',q/'build-source255-best-of-005/source-build.json']
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):sha(p) for p in inputs},'report_sha256':sha(d/'report.json')},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'crash_sha256':sha(copy)}))
