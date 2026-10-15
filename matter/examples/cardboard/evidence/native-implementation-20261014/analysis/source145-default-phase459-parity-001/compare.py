from pathlib import Path
import hashlib,json,importlib.util
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/source145-default-phase459-parity-001';O.mkdir(exist_ok=False)
L=B/'solver/cgls-state-sidecar-observer-455';R=B/'solver/post-release-default-zero-parity-459';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=B/'analysis/native450-cgls-candidate-audit-tool-001/audit_cgls_snapshot.py';a=load(T,'binding459')
lp=json.loads((L/'prediction.json').read_text());rp=json.loads((R/'prediction.json').read_text());ls=Path(lp['source_snapshot']);rs=Path(rp['source_snapshot']);paths=[(L,R),(ls,rs),(Path(lp['runtime_dir']),Path(rp['runtime_dir']))]
def norm(s):
 for left,right in paths:s=s.replace(str(left),'<'+left.name+'>').replace(str(right),'<'+left.name+'>')
 return s
for run,snap in [(L,ls),(R,rs)]:
 m,_=a.load_audit_module(snap);_,receipt,_,_=a.bind_run(run,snap,m);assert receipt['exit_code']==2 and not receipt['timed_out'] and receipt['source_inputs_unchanged']
lf={str(x.relative_to(L/'output')):x for x in (L/'output').rglob('*') if x.is_file()};rf={str(x.relative_to(R/'output')):x for x in (R/'output').rglob('*') if x.is_file()};assert lf.keys()==rf.keys();rows=[]
for k in sorted(lf):
 same=lf[k].read_bytes()==rf[k].read_bytes();row={'path':k,'raw_equal':same,'left_sha256':sha(lf[k]),'right_sha256':sha(rf[k])}
 if not same:
  assert k=='manifest.json',k
  lv=json.loads(norm(lf[k].read_text()));rv=json.loads(norm(rf[k].read_text()));found=[]
  def prune(v,loc=''):
   if isinstance(v,dict):
    if 'post_release_relaxation_steps' in v:assert v['post_release_relaxation_steps']==0;found.append(loc+'/post_release_relaxation_steps');del v['post_release_relaxation_steps']
    for kk,vv in v.items():prune(vv,loc+'/'+kk)
   elif isinstance(v,list):
    for i,vv in enumerate(v):prune(vv,loc+'/'+str(i))
  prune(rv);assert len(found)==1,found;assert lv==rv,k;row['metadata_only_default_zero_field']=found
 rows.append(row)
payloads={}
for k in ['base-cgls.bin','base-cgls.bin.state.bin','first-status5.bin','stdout.log']:
 assert (L/k).read_bytes()==(R/k).read_bytes(),k;payloads[k]={'bytes':(L/k).stat().st_size,'sha256':sha(L/k),'byte_equal':True}
for k in ['base-cgls.bin.state.json','first-status5.json']:
 assert json.loads(norm((L/k).read_text()))==json.loads(norm((R/k).read_text())),k
lt=[norm(x) for x in (L/'stderr.log').read_text().splitlines() if not x.startswith('cardboard_step_timing=')];rt=[norm(x) for x in (R/'stderr.log').read_text().splitlines() if not x.startswith('cardboard_step_timing=')];assert lt==rt,'non-timing stderr differs'
lsf=json.loads((ls/'frozen-source.json').read_text())['source_sha256'];rsf=json.loads((rs/'frozen-source.json').read_text())['source_sha256'];changes=[k for k in lsf if lsf[k]!=rsf[k]];assert set(changes)=={'matter/tools/cardboard_probe.mm','matter/tools/cardboard_convergence.py','matter/tools/cardboard_resolution_study.py','matter/tools/test_cardboard_convergence.py','matter/tools/test_cardboard_resolution_study.py'}
for snap,files in [(ls,lsf),(rs,rsf)]:
 for k,v in files.items():assert sha(snap/k)==v,k
report={'schema':'numi.cardboard.source145-default-zero-full-attempt-parity.v1','parity_pass':True,'source_changes':changes,'source_files_verified_per_snapshot':len(lsf),'output_files':rows,'payloads':payloads,'non_timing_stderr_lines':len(lt),'production_trace_counts':{x:sum(line.startswith(x+'=') for line in lt) for x in ['fem_residual_armijo_iteration','fem_true_residual_cycle','fem_base_cgls_direction']},'state_marker_and_status_manifest_equal_after_explicit_path_mapping':True,'limits':['Both attempts fail and roll back control1 after one accepted control.','Default zero phase was compared; no new post-release phase was executed.','No complete coupon, mesh/time convergence, calibrated corrugated failure, assembly, crush or impact qualification.']}
(O/'compare.py').write_bytes(Path(__file__).read_bytes());(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'manifest.json').write_text(json.dumps({'inputs':{str(p):sha(p) for p in [T,L/'prediction.json',L/'receipt.json',R/'prediction.json',R/'receipt.json',L/'stderr.log',R/'stderr.log']},'files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}},indent=2)+'\n');print('PASS',len(rows),report['production_trace_counts'])
