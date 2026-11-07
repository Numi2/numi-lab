import hashlib,json,os,pathlib,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2];source=root/'tests/contact/fem_p2_dynamics.py'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
cmd=[sys.executable,str(source),'check'];start=time.monotonic()
with (p/'checks.json').open('w') as out,(p/'checks.stderr').open('w') as err:
 try:status=subprocess.run(cmd,stdout=out,stderr=err,env=env,timeout=120).returncode
 except subprocess.TimeoutExpired:status='timeout'
m={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':sha(source),'cmd':cmd,'exit':status,'seconds':time.monotonic()-start,'output_sha256':sha(p/'checks.json')}
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m,flush=True)
if status!=0:raise SystemExit('execution failure retained')
d=json.loads((p/'checks.json').read_text());mass=d['mass']
m['pass']=mass['relative_error']<=1e-10 and mass['symmetry_l2']<=1e-12 and mass['min_eigenvalue']>0 and mass['freefall_relative']<=1e-9
for name in ('initial','active'):
 x=d[name];m['pass'] &= x['min_gap_m']>0 and x['min_jacobian']>0 and x['energy_relative']<=2e-6 and x['contact_power_relative']<=2e-6
m['pass'] &= max(d['quadrature'].values())<=.0025
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('pass',m['pass'],flush=True)
if not m['pass']:raise SystemExit('numerical failure retained')
