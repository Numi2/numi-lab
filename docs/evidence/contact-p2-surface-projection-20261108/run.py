import hashlib,json,os,pathlib,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
source=root/'tests/contact/fem_p2_surface_projection.py';model=root/'tests/contact/fem_p2_dynamics.py'
a=root/'docs/evidence/contact-p2-n4-order18-20261108/fine.json';b=root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
cmd=[sys.executable,str(source),str(a),str(b)];start=time.monotonic()
with (p/'projection.json').open('w') as out,(p/'projection.stderr').open('w') as err:
 try:status=subprocess.run(cmd,stdout=out,stderr=err,env=env,timeout=180).returncode
 except subprocess.TimeoutExpired:status='timeout'
m={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':sha(source),'model_sha256':sha(model),'n4_sha256':sha(a),'n8_sha256':sha(b),
 'cmd':cmd,'exit':status,'timeout_s':180,'seconds':time.monotonic()-start,'output_sha256':sha(p/'projection.json')}
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m,flush=True)
if status!=0:raise SystemExit('execution failure retained')
