import hashlib,json,os,pathlib,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
source=root/'tests/contact/fem_p2_quadrature_sweep.py';model=root/'tests/contact/fem_p2_dynamics.py'
inp=root/'docs/evidence/contact-p2-n2-trajectory-corrected-20261108/fine.json'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
cmd=[sys.executable,str(source),str(inp)];start=time.monotonic()
with (p/'sweep.json').open('w') as out,(p/'sweep.stderr').open('w') as err:
 try:status=subprocess.run(cmd,stdout=out,stderr=err,env=env,timeout=180).returncode
 except subprocess.TimeoutExpired:status='timeout'
m={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':sha(source),'model_sha256':sha(model),'input_sha256':sha(inp),'cmd':cmd,'exit':status,
 'seconds':time.monotonic()-start,'output_sha256':sha(p/'sweep.json')}
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m,flush=True)
if status!=0:raise SystemExit('failure retained')
