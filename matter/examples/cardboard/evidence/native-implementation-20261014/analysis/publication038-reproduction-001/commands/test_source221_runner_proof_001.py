from pathlib import Path
import json,hashlib,subprocess,datetime,sys,copy
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-qualification-checkpoint-proof-221-003';b=q/'build-source221-checkpoint-proof-003';d=q/'analysis/source221-runner-proof-cpu-tests-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text())
assert bm['success'] and bm['source_sha256']==f['source_sha256']
assert all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
d.mkdir(exist_ok=False)
donor=q/'solver/fresh-source210-strict-inner-full-coupon-545';origpred=json.loads((donor/'prediction.json').read_text());origrecord=json.loads((donor/'native-environment.json').read_text())
binary=b/'numi-matter-cardboard-probe';results=[]
def dump(j):return json.dumps(j,sort_keys=True,indent=2)+'\n'
def digest(j):return hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def case(name,change,expected,reseal=True):
 folder=d/name;folder.mkdir();pred=copy.deepcopy(origpred);record=copy.deepcopy(origrecord)
 change(pred,record)
 if reseal:
  contract=pred['effective_native_environment_contract']
  contract['sha256']=digest({k:v for k,v in contract.items() if k!='sha256'})
  pred['effective_native_environment_contract_sha256']=contract['sha256']
  record['contract']=copy.deepcopy(contract);record['contract_sha256']=contract['sha256']
  pred['environment']={k:v['value'] for k,v in contract['effective'].items() if v['set']}
 data=dump(pred);(folder/'prediction.json').write_text(data)
 record['prediction_sha256']=hashlib.sha256(data.encode()).hexdigest()
 (folder/'native-environment.json').write_text(dump(record))
 argv=[str(binary),'--verify-coupon-source-runner',str(folder)]
 r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
 (folder/'stdout.log').write_bytes(r.stdout);(folder/'stderr.log').write_bytes(r.stderr)
 results.append({'name':name,'argv':argv,'expected_accept':expected,'exit_code':r.returncode,'pass':(r.returncode==0)==expected,'stdout_sha256':sha(folder/'stdout.log'),'stderr_sha256':sha(folder/'stderr.log')})
def older(n):
 def change(p,r):
  remove=['NM_FEM_STRICT_INNER_FORCING','NM_FEM_BASE_CGLS_DAMPING_KG','NM_CONTACT_NODE_INCIDENCE_SORT','NM_FEM_BASE_CGLS_WARM_START_REFINE'][:53-n]
  c=p['effective_native_environment_contract']
  for key in remove:c['keys'].remove(key);del c['effective'][key]
 return change
case('valid-53-current',lambda p,r:None,True)
for n in [52,51,50,49]:case('valid-'+str(n)+'-legacy',older(n),True)
def badkey(p,r):
 c=p['effective_native_environment_contract'];c['keys'][0]='NM_A_UNREGISTERED';c['effective']['NM_A_UNREGISTERED']=c['effective'].pop('NM_CONTACT_NODE_INCIDENCE_SORT')
case('reject-signed-but-unregistered-key',badkey,False)
case('reject-nonboolean-set',lambda p,r:p['effective_native_environment_contract']['effective']['NM_FEM_STRICT_INNER_FORCING'].update({'set':1}),False)
case('reject-not-scrubbed',lambda p,r:r['launch'].update({'native_controls_scrubbed':False}),False)
case('reject-mixed-record-schema',lambda p,r:r.update({'schema':'numi.cardboard.native-diagnostic-environment-run.v1'}),False)
def badpath(p,r):
 p['effective_native_environment_contract']['effective']['NM_FEM_OPERATOR_FD_CAPTURE_PATH']={'set':True,'value':str(d/'unapproved-output')}
case('reject-qualification-with-output',badpath,False)
case('reject-wrong-contract-sha',lambda p,r:p.update({'effective_native_environment_contract_sha256':'0'*64}),False,False)
# Also exercise the unmodified real retained records, not just constructed ones.
r=subprocess.run([str(binary),'--verify-coupon-source-runner',str(donor)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
(d/'actual545.stdout').write_bytes(r.stdout);(d/'actual545.stderr').write_bytes(r.stderr)
results.append({'name':'actual-retained545','exit_code':r.returncode,'expected_accept':True,'pass':r.returncode==0})
r2=subprocess.run([str(b/'numi-matter-cardboard-coupon-resume-check')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
(d/'coupon-resume-check.stdout').write_bytes(r2.stdout);(d/'coupon-resume-check.stderr').write_bytes(r2.stderr)
report={'schema':'numi.cardboard.source221-runner-proof-cpu-tests.v1','cases':results,'all_pass':all(r['pass'] for r in results) and r2.returncode==0,'existing_resume_check_exit_code':r2.returncode,'frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'actual_source_runner_prediction_sha256':sha(donor/'prediction.json'),'actual_source_runner_record_sha256':sha(donor/'native-environment.json'),'gpu_execution':False,'native_runtime_created':False,'historical_files_unchanged':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(dump(report))
(d/'manifest.json').write_text(dump({'files':{str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}}));print(dump(report));sys.exit(int(not report['all_pass']))
