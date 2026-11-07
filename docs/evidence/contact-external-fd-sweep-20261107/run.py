import hashlib,json,os,pathlib,subprocess
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
script=root/'tests/contact/fem_external_scikit_probe.py';native=root/'docs/evidence/contact-nodal-parity-20261107/native-h31.25us.json'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
manifest={'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':sha(script),'input_sha256':sha(native),'samples':[]}
for n,h in enumerate(['4e-8','2e-8','1e-8','5e-9','2.5e-9','1.25e-9','6.25e-10']):
 state=p/f'{n}.state';env=dict(os.environ,NUMI_EXTERNAL_FD_EPS=h)
 run=subprocess.run(['/tmp/numi-fem-py313b-20261107/bin/python',str(script),str(native),'319',str(state)],env=env,capture_output=True,text=True)
 (p/f'{n}.json').write_text(run.stdout);(p/f'{n}.stderr').write_text(run.stderr)
 if run.returncode:raise SystemExit('retained execution failure')
 a=json.loads(run.stdout);r=a['analytic_energy_directional_derivative']
 manifest['samples'].append({'epsilon_m':float(h),'energy_fd_relative':abs(a['energy_directional_derivative']-r)/abs(r),
 'tangent_fd_relative':a['fd_tangent_l2_relative']})
 (p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(manifest['samples'][-1],flush=True)
