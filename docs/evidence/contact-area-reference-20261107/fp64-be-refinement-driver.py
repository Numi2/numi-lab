import json,subprocess,pathlib,time
e=pathlib.Path('/Users/numi/numi-lab-contact-reference-20261107/docs/evidence/contact-area-reference-20261107')
exe='/Users/numi/numi-lab-contact-area-evidence-20261107/fem-plane-fp64-backward-euler'
ref=json.loads((e/'fp64-fem-h0.5us.json').read_text())
results=[]
for h in [31.25e-6,15.625e-6,7.8125e-6,3.90625e-6,1.953125e-6,.9765625e-6]:
 started=time.monotonic()
 r=subprocess.run([exe,str(h),'.01'],capture_output=True,text=True)
 name='fp64-be-refine-'+str(h)
 (e/(name+'.json')).write_text(r.stdout)
 (e/(name+'.stderr')).write_text(r.stderr)
 row={'h':h,'cpu_wall_seconds':time.monotonic()-started,'returncode':r.returncode}
 if r.returncode==0:
  x=json.loads(r.stdout)
  row.update(force_error=max(abs(a-b) for a,b in zip(x['force_bins_N'],ref['force_bins_N']))/max(ref['force_bins_N']),
   impulse_error=abs(x['impulse_Ns']-ref['impulse_Ns'])/abs(ref['impulse_Ns']),
   max_nonlinear_residual_m=x['max_nonlinear_residual_m'], max_newton_iterations=x['max_newton_iterations'])
 results.append(row)
 (e/'fp64-be-refinement-summary.json').write_text(json.dumps(results,indent=2)+'\n')
 print(row,flush=True)
 if r.returncode!=0 or (row['force_error']<=.0025 and row['impulse_error']<=.0025):break
