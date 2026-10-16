from pathlib import Path
import hashlib, importlib.util,json
B=Path('/Users/home/cardboard-implementation-20261014')
O=B/'analysis/source154-default64-native466-parity-001'
O.mkdir(exist_ok=False)
L=B/'solver/cgls-block-right-primary48-463'; R=B/'solver/cgls-newton128-default64-parity-466'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'compare.py').write_bytes(Path(__file__).read_bytes())
try:
 T=B/'analysis/native450-cgls-candidate-audit-tool-001/audit_cgls_snapshot.py'
 spec=importlib.util.spec_from_file_location('binding466',T); audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 lp=json.loads((L/'prediction.json').read_text());rp=json.loads((R/'prediction.json').read_text())
 ls=Path(lp['source_snapshot']);rs=Path(rp['source_snapshot'])
 for run,snap in [(L,ls),(R,rs)]:
  module,_=audit.load_audit_module(snap);_,receipt,_,_=audit.bind_run(run,snap,module)
  assert receipt['exit_code']==2 and not receipt['timed_out'] and receipt['source_inputs_unchanged']
 paths=[(L,R),(ls,rs),(Path(lp['runtime_dir']),Path(rp['runtime_dir']))]
 left_state=json.loads((L/'base-cgls.bin.state.json').read_text());right_state=json.loads((R/'base-cgls.bin.state.json').read_text())
 oldfp=left_state['device_program_fingerprint_u64_hex'];newfp=right_state['device_program_fingerprint_u64_hex']
 assert left_state['world_fingerprint_u64_hex']==right_state['world_fingerprint_u64_hex']
 normalizations=[]
 def ns(v):
  for left,right in paths:v=v.replace(str(left),'<'+left.name+'>').replace(str(right),'<'+left.name+'>')
  return v
 def cj(a,b,c):
  def visit(x,y,p):
   if isinstance(x,dict) and isinstance(y,dict):
    assert x.keys()==y.keys(),(c,p,x.keys(),y.keys())
    for k in x:visit(x[k],y[k],p+'/'+k)
   elif isinstance(x,list) and isinstance(y,list):
    assert len(x)==len(y),(c,p)
    for i,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,p+'/'+str(i))
   elif isinstance(x,str) and isinstance(y,str):
    if p in ('/device_program_fingerprint_u64_hex','/device_program_fingerprint'):
     assert (x,y) in ((oldfp,newfp),(str(int(oldfp,16)),str(int(newfp,16)))),(c,p,x,y)
     if x!=y:normalizations.append({'context':c,'path':p,'kind':'independently_bound_program_identity'})
    else:assert ns(x)==ns(y),(c,p,x,y)
   else:assert type(x)==type(y) and x==y,(c,p,x,y)
  visit(a,b,'')
 lf={str(p.relative_to(L/'output')):p for p in (L/'output').rglob('*') if p.is_file()}
 rf={str(p.relative_to(R/'output')):p for p in (R/'output').rglob('*') if p.is_file()}
 assert lf.keys()==rf.keys();rows=[]
 for k,p in sorted(lf.items()):
  q=rf[k];same=p.read_bytes()==q.read_bytes()
  if not same:
   assert Path(k).name=='manifest.json',k;cj(json.loads(p.read_text()),json.loads(q.read_text()),'output/'+k)
  rows.append({'path':k,'raw_equal':same,'left_sha256':sha(p),'right_sha256':sha(q)})
 payloads={}
 for k in ['base-cgls.bin','base-cgls.bin.state.bin','first-status5.bin','stdout.log']:
  assert (L/k).read_bytes()==(R/k).read_bytes(),k
  payloads[k]={'bytes':(L/k).stat().st_size,'sha256':sha(L/k),'raw_equal':True}
 for k in ['base-cgls.bin.state.json','first-status5.json']:cj(json.loads((L/k).read_text()),json.loads((R/k).read_text()),k)
 ll=[x for x in (L/'stderr.log').read_text().splitlines() if not x.startswith('cardboard_step_timing=')]
 rl=[x for x in (R/'stderr.log').read_text().splitlines() if not x.startswith('cardboard_step_timing=')]
 assert len(ll)==len(rl),(len(ll),len(rl))
 for i,(x,y) in enumerate(zip(ll,rl)):
  if ns(x)==ns(y):continue
  lk,lv=x.split('=',1);rk,rv=y.split('=',1);assert lk==rk
  cj(json.loads(lv),json.loads(rv),'stderr/'+str(i)+'/'+lk)
 lfs=json.loads((ls/'frozen-source.json').read_text())['source_sha256'];rfs=json.loads((rs/'frozen-source.json').read_text())['source_sha256']
 assert lfs.keys()==rfs.keys();changes=[k for k in lfs if lfs[k]!=rfs[k]]
 assert set(changes)=={'matter/src/runtime.mm','matter/tools/cardboard_probe.mm'},changes
 for s,entries in [(ls,lfs),(rs,rfs)]:
  for k,h in entries.items():assert sha(s/k)==h,k
 # Build topology is bound by each run's verified receipt; record it explicitly.
 report={'schema':'numi.cardboard.source154-default64-native466-parity.v1','parity_pass':True,'left_case':str(L),'right_case':str(R),'source_changes':changes,'source_files_verified_per_snapshot':len(lfs),'output_files':rows,'payloads':payloads,'normalizations':normalizations,'execution_identity':{'left':oldfp,'right':newfp,'treated_as_equal':oldfp==newfp},'non_timing_stderr_lines':len(ll),'production_trace_counts':{k:sum(x.startswith(k+'=') for x in ll) for k in ['fem_residual_armijo_iteration','fem_true_residual_cycle','fem_base_cgls_direction']},'build_topology':{'left':'root CMake project, Matter plus MetalWorld linked artifacts','right':'standalone matter CMake project, three bound artifacts','comparison':'measured complete attempted-history parity, not assumed binary or topology equivalence'},'limits':['Both attempts fail and roll back control1 after one accepted control.','This validates existing numeric/default64 behavior only; new all-control/128 behavior is not evaluated.','No complete coupon, numerical convergence, calibration, assembly, crush or impact qualification.']}
 (O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 (O/'manifest.json').write_text(json.dumps({'inputs':{str(p):sha(p) for p in [T,L/'prediction.json',L/'receipt.json',R/'prediction.json',R/'receipt.json',L/'stderr.log',R/'stderr.log']},'files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}},indent=2)+'\n')
 print(json.dumps({'parity_pass':True,'outputs':len(rows),'trace_counts':report['production_trace_counts'],'normalization_count':len(normalizations)}))
except Exception as e:
 (O/'failure.json').write_text(json.dumps({'error_type':type(e).__name__,'error':str(e),'verdict':'unadmitted; comparison did not complete'},indent=2)+'\n');raise
