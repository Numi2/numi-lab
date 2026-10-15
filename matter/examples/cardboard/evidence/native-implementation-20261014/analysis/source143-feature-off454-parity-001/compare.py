from pathlib import Path
import importlib.util,json,hashlib,sys
B=Path('/Users/home/cardboard-implementation-20261014');O=Path(__file__).resolve().parent
L=B/'solver/bounded-cgls64-exact-state-450';R=B/'solver/cgls-primary-feature-off-parity-454'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=B/'analysis/native450-cgls-candidate-audit-tool-001/audit_cgls_snapshot.py';a=load(T,'binding454')
P=B/'analysis/native450-capture-off-parity-tool-001/audit_capture_observer_parity.py';p=load(P,'parity454')
lp=json.loads((L/'prediction.json').read_text());rp=json.loads((R/'prediction.json').read_text());ls=Path(lp['source_snapshot']);rs=Path(rp['source_snapshot'])
lmod,_=a.load_audit_module(ls);rmod,_=a.load_audit_module(rs)
_,lr,_,lb=a.bind_run(L,ls,lmod);_,rr,_,rb=a.bind_run(R,rs,rmod)
assert lp['executed_build_verified'] and rp['executed_build_verified'];assert lr['exit_code']==rr['exit_code']==2
assert 'NM_FEM_BASE_CGLS_PRIMARY_AFTER' not in rp['environment']
lc,la,le,li=p.command_parts(B/'native-450-command.json');rc,ra,re,ri=p.command_parts(B/'native-454-command.json')
paths=[(L,R),(ls,rs),(Path(lp['runtime_dir']),Path(rp['runtime_dir']))]
def normalize(s):
 for left,right in paths:s=s.replace(str(left),'<'+str(left.name)+'>').replace(str(right),'<'+str(left.name)+'>')
 return s
assert [normalize(x) for x in la]==[normalize(x) for x in ra]
assert {k:normalize(v) for k,v in le.items()}=={k:normalize(v) for k,v in re.items()}
assert {k:normalize(v) for k,v in lp['environment'].items()}=={k:normalize(v) for k,v in rp['environment'].items()}
lf=json.loads((ls/'frozen-source.json').read_text())['source_sha256'];rf=json.loads((rs/'frozen-source.json').read_text())['source_sha256'];assert lf.keys()==rf.keys()
diff={k:{'old':lf[k],'new':rf[k]} for k in lf if lf[k]!=rf[k]};assert set(diff)=={'matter/src/runtime.mm','matter/src/metal/base_cgls_runtime.metalinc'}
for S,F in [(ls,lf),(rs,rf)]:
 for k,d in F.items():assert sha(S/k)==d,k
raw=p.exact_tree(L/'output',R/'output');rawdiff=[x['path'] for x in raw if not x['equal']];assert rawdiff==['manifest.json','rejected-conditioning-000002/manifest.json'],rawdiff
lm=json.loads((L/'output/rejected-conditioning-000002/manifest.json').read_text());rm=json.loads((R/'output/rejected-conditioning-000002/manifest.json').read_text());oldfp=lm.pop('device_program_fingerprint');newfp=rm.pop('device_program_fingerprint');assert lm==rm
lmain=json.loads(normalize((L/'output/manifest.json').read_text()));rmain=json.loads(normalize((R/'output/manifest.json').read_text()));assert lmain==rmain
stl=json.loads((L/'first-status5.json').read_text());strr=json.loads((R/'first-status5.json').read_text());oldhex=stl.pop('device_program_fingerprint');newhex=strr.pop('device_program_fingerprint');assert int(oldfp)==int(oldhex,16) and int(newfp)==int(newhex,16)
stl['payload']['path']=normalize(stl['payload']['path']);strr['payload']['path']=normalize(strr['payload']['path']);assert stl==strr
assert (L/'first-status5.bin').read_bytes()==(R/'first-status5.bin').read_bytes()
assert (L/'base-cgls.bin').read_bytes()==(R/'base-cgls.bin').read_bytes();assert (L/'stdout.log').read_bytes()==(R/'stdout.log').read_bytes()
keep=lambda q:[normalize(x) for x in (q/'stderr.log').read_text().splitlines() if not x.startswith('cardboard_step_timing=')]
ltrace,rtrace=keep(L),keep(R);assert ltrace==rtrace,'non-timing diagnostics differ'
counts={pre:len(p.parse_prefixed(L/'stderr.log',pre)) for pre in ['fem_residual_armijo_iteration=','fem_true_residual_cycle=','fem_base_cgls_direction=']}
report={'schema':'numi.cardboard.source143_feature_off_full_attempted_history_parity.v1','parity_pass':True,'classification':'default-off source/build change parity across the full attempted trajectory; both coupon attempts failed, not a completed coupon','run450':{'path':str(L),'receipt_sha256':sha(L/'receipt.json'),'binding':lb},'run454':{'path':str(R),'receipt_sha256':sha(R/'receipt.json'),'binding':rb},'source_changes':diff,'all_source_files_verified':len(lf),'physical_and_solver_arguments_equal_after_explicit_path_mapping':True,'environment_equal_after_run_path_mapping':True,'physical_output_files':len(raw),'raw_byte_equal_output_files':len(raw)-len(rawdiff),'metadata_only_output_differences':rawdiff,'metadata_exclusions':{'output_manifest':'three material source paths only; file hashes and all other values equal','conditioning_manifest':'device_program_fingerprint only','status5_manifest':'payload path and device_program_fingerprint only','fingerprints':{'450_decimal':oldfp,'454_decimal':newfp,'450_hex':oldhex,'454_hex':newhex},'stderr':'only wall/GPU timing rows excluded, run/source/runtime paths normalized; all other rows exactly equal'},'full_numeric_trace_counts':counts,'non_timing_stderr_lines':len(ltrace),'CGLS_snapshot_byte_identical':True,'status5_payload_byte_identical':True,'stdout_byte_identical':True,'rollback_geometry_byte_identical':True,'source_builds_verified_by_native_wrapper':True,'physical_validation':False,'limits':['Feature-on453 still fails control1 after64 Newton iterations.','No coupon convergence, nonlinear contact, material calibration, box assembly or crush/drop qualification.']}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');inputs=[T,P,B/'native-450-command.json',B/'native-454-command.json',L/'prediction.json',L/'receipt.json',R/'prediction.json',R/'receipt.json',L/'stderr.log',R/'stderr.log'];(O/'manifest.json').write_text(json.dumps({'inputs':{str(x):sha(x) for x in inputs},'files':{'compare.py':sha(Path(__file__)),'report.json':sha(O/'report.json')}},indent=2)+'\n');print('PASS',counts,len(raw),len(ltrace))
