import hashlib,json,os,pathlib,subprocess,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
prog=root/'tests/contact/fem_external_dynamic_diagnostic.py';probe=root/'tests/contact/fem_external_scikit_n4.py'
native=root/'docs/evidence/contact-nodal-parity-20261107/native-h31.25us.json'
ref=root/'docs/evidence/contact-mesh-reference-20261107/n4-h5e-07.json'
n2=root/'docs/evidence/contact-external-dynamics-20261107/fine.json'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
m={'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'dynamic_sha256':sha(prog),'assembly_sha256':sha(probe),'native_sha256':sha(native),'reference_sha256':sha(ref),
 'n2_external_sha256':sha(n2),'python':'/tmp/numi-fem-py313b-20261107/bin/python','timeout_seconds':300,'runs':[]}
for variant in ('coarse','fine'):
 cmd=[m['python'],str(prog),str(probe),str(native),variant];now=time.monotonic()
 with (p/(variant+'.json')).open('w') as out,(p/(variant+'.stderr')).open('w') as err:
  try:status=subprocess.run(cmd,stdout=out,stderr=err,env=env,timeout=300).returncode
  except subprocess.TimeoutExpired:status='timeout'
 m['runs'].append({'variant':variant,'argv':cmd,'exit':status,'elapsed_seconds':time.monotonic()-now,'output_sha256':sha(p/(variant+'.json'))})
 (p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(m['runs'][-1],flush=True)
 if status!=0:raise SystemExit('retained failure; stopping')
