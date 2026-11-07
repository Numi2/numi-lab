import contextlib,hashlib,io,json,os,pathlib,runpy,subprocess,sys,time,numpy as np
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
native=root/'docs/evidence/contact-nodal-parity-20261107/native-h31.25us.json'
dynamic=root/'tests/contact/fem_external_dynamic_residual_only.py'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
manifest={'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'native_sha256':sha(native),'dynamic_sha256':sha(dynamic),'mesh':{}}
def module(path):
 sys.argv=[str(path),str(native),'150','/tmp/numi-residual-parity-state.txt']
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): return runpy.run_path(str(path))
for n in (2,4):
 old=root/('tests/contact/fem_external_scikit_probe.py' if n==2 else 'tests/contact/fem_external_scikit_n4.py')
 fast=root/f'tests/contact/fem_external_scikit_n{n}_residual_only.py'
 a,b=module(old),module(fast);ref=a['ref'];area=a['area'];rng=np.random.default_rng(81)
 vec=rng.normal(size=ref.shape);vec/=np.max(np.abs(vec))
 if n==2:
  data=json.loads(native.read_text())
  states=[('retained_'+str(i),np.array(data['traces'][i]['node_state'])[:,:3]) for i in (0,150,200,319)]
 else:
  active=ref+np.array([0.,0.,-.000105])
  perturb=rng.normal(size=ref.shape)*1e-7
  states=[('initial',ref),('uniform_active',active),('nonuniform_active',active+perturb)]
 checks=[]
 for name,x in states:
  ea,ra,ta=a['energy_residual_tangent'](x,vec)
  eb,rb,tb=b['energy_residual_tangent'](x,vec,compute_tangent=False)
  _,_,tbcheck=b['energy_residual_tangent'](x,vec,compute_tangent=True)
  e=abs(ea-eb);r=np.linalg.norm(ra-rb)/max(np.linalg.norm(ra),1e-30)
  t=np.linalg.norm(ta-tbcheck)/max(np.linalg.norm(ta),1e-30)
  gap=np.min(x[area>0,2]);checks.append({'state':name,'energy_abs':e,'residual_rel':r,'tangent_rel':t,'min_gap':gap,
   'pass':bool(e<=1e-12 and r<=1e-12 and t<=1e-12 and gap>0)})
 manifest['mesh'][str(n)]={'original_sha256':sha(old),'residual_source_sha256':sha(fast),'state_checks':checks}
 (p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('state_checks',n,checks,flush=True)
 if not all(x['pass'] for x in checks):raise SystemExit('state failure retained')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for n,timeout in ((2,120),(4,180)):
 fast=root/f'tests/contact/fem_external_scikit_n{n}_residual_only.py'
 original=root/('docs/evidence/contact-external-dynamics-20261107/fine.json' if n==2 else 'docs/evidence/contact-external-n4-20261107/fine.json')
 cmd=[sys.executable,str(dynamic),str(fast),str(native),'fine'];now=time.monotonic()
 with (p/f'n{n}-fine.json').open('w') as out,(p/f'n{n}-fine.stderr').open('w') as err:
  try:status=subprocess.run(cmd,stdout=out,stderr=err,env=env,timeout=timeout).returncode
  except subprocess.TimeoutExpired:status='timeout'
 result={'exit':status,'timeout':timeout,'elapsed_seconds':time.monotonic()-now,'original_output_sha256':sha(original),'output_sha256':sha(p/f'n{n}-fine.json')}
 manifest['mesh'][str(n)]['run']=result
 (p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if status!=0:raise SystemExit('execution failure retained')
 a=json.loads(original.read_text());b=json.loads((p/f'n{n}-fine.json').read_text())
 force=max(abs(x-y) for x,y in zip(a['force_bins_N'],b['force_bins_N']))/max(a['force_bins_N'])
 imp=abs(a['impulse_Ns']-b['impulse_Ns'])/abs(a['impulse_Ns'])
 work=abs(a['contact_work_J']-b['contact_work_J'])
 balance=b['momentum_error_Ns']/abs(b['impulse_Ns'])
 parity={'force_bin_relative':force,'impulse_relative':imp,'contact_work_abs_J':work,
  'min_sampled_gap_m':b['min_sampled_gap_m'],'momentum_relative':balance,
  'pass':force<=1e-10 and imp<=1e-10 and work<=1e-10 and b['min_sampled_gap_m']>0 and balance<=.0025}
 manifest['mesh'][str(n)]['trajectory_parity']=parity
 (p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('trajectory',n,result,parity,flush=True)
 if not parity['pass']:raise SystemExit('parity failure retained')
