import hashlib,json,pathlib,subprocess,numpy as np
p=pathlib.Path(__file__).resolve().parent
root=p.parents[2];py='/tmp/numi-fem-py313b-20261107/bin/python';exe='/tmp/numi_external_probe_20261107'
source=root/'tests/contact/fem_external_scikit_probe.py'
cpp=root/'tests/contact/fem_external_reference_probe.cpp'
native=root/'docs/evidence/contact-nodal-parity-20261107/native-h31.25us.json'
def digest(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'python_version':subprocess.check_output([py,'--version'],text=True).strip(),
 'dependencies':subprocess.check_output([py,'-m','pip','freeze'],text=True).splitlines(),
 'source_sha256':digest(source),'cpp_sha256':digest(cpp),'exe_sha256':digest(exe),
 'model_sha256':digest(root/'tests/contact/fem_plane_surface_diagnostic.cpp'),'native_sha256':digest(native),'checks':[]}
for step in (0,150,200,319):
 prefix=p/f'step-{step}';state=str(prefix)+'.state'
 out=subprocess.run([py,str(source),str(native),str(step),state],capture_output=True,text=True,check=True)
 pathlib.Path(str(prefix)+'.scikit.json').write_text(out.stdout)
 pathlib.Path(str(prefix)+'.scikit.stderr').write_text(out.stderr)
 with open(state) as f:
  cp=subprocess.run([exe],stdin=f,capture_output=True,text=True,check=True)
 pathlib.Path(str(prefix)+'.cpp.txt').write_text(cp.stdout)
 j=json.loads(out.stdout);c=np.loadtxt(str(prefix)+'.cpp.txt',skiprows=1)
 energy=float(cp.stdout.splitlines()[0]);r=np.array(j['residual']);t=np.array(j['tangent'])
 re=np.linalg.norm(r-c[:,:3])/max(np.linalg.norm(r),1e-30)
 te=np.linalg.norm(t-c[:,3:])/max(np.linalg.norm(t),1e-30)
 ee=abs(j['energy_J']-energy)/max(abs(energy),1e-30)
 fd=abs(j['energy_directional_derivative']-j['analytic_energy_directional_derivative'])/max(abs(j['analytic_energy_directional_derivative']),1e-30)
 controls=j['analytic_controls']
 entry={'step':step,'active_nodes':j['active_nodes'],'energy_relative':ee,'residual_l2_relative':re,
  'tangent_l2_relative':te,'external_energy_fd_relative':fd,
  'external_tangent_fd_l2_relative':j['fd_tangent_l2_relative'],'analytic_controls':controls,
  'pass':ee<=1e-8 and re<=1e-8 and te<=1e-6 and fd<=2e-6 and j['fd_tangent_l2_relative']<=2e-6
   and controls['affine_energy_absolute_error_J']<=1e-12
   and controls['affine_element_density_max_absolute_error_Pa']<=1e-8
   and controls['uniform_contact_energy_absolute_error_J']<=1e-12
   and controls['uniform_contact_total_residual_error_N']<=1e-9}
 manifest['checks'].append(entry)
 (p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(entry),flush=True)
 if not entry['pass']:raise SystemExit('failed check retained; stopping')
