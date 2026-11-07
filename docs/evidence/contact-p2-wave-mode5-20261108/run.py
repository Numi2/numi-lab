import hashlib,json,os,pathlib,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2];source=root/'tests/contact/fem_p2_linear_wave_mode5.py'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
cmd=[sys.executable,str(source)];start=time.monotonic()
with (p/'wave.json').open('w') as out,(p/'wave.stderr').open('w') as err:
 try:status=subprocess.run(cmd,stdout=out,stderr=err,env=env,timeout=180).returncode
 except subprocess.TimeoutExpired:status='timeout'
m={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_sha256':sha(source),
 'cmd':cmd,'timeout_s':180,'exit':status,'elapsed_s':time.monotonic()-start,'output_sha256':sha(p/'wave.json')}
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m,flush=True)
if status!=0:raise SystemExit('failure retained')
