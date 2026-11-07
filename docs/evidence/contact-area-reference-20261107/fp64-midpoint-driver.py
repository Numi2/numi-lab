import json,subprocess,pathlib,time
e=pathlib.Path('/Users/numi/numi-lab-contact-reference-20261107/docs/evidence/contact-area-reference-20261107')
exe='/Users/numi/numi-lab-contact-area-evidence-20261107/fem-plane-fp64-midpoint'
ref=json.loads((e/'fp64-fem-h0.5us.json').read_text())
rows=[]
for h in [500e-6,250e-6,125e-6,62.5e-6,31.25e-6,15.625e-6,7.8125e-6,3.90625e-6]:
 t=time.monotonic();r=subprocess.run([exe,str(h),'.01'],capture_output=True,text=True)
 name='fp64-midpoint-h'+str(h)
 (e/(name+'.json')).write_text(r.stdout);(e/(name+'.stderr')).write_text(r.stderr)
 row={'h':h,'cpu_seconds':time.monotonic()-t,'returncode':r.returncode}
 if r.returncode==0:
  x=json.loads(r.stdout)
  row.update(force_error=max(abs(a-b) for a,b in zip(x['force_bins_N'],ref['force_bins_N']))/max(ref['force_bins_N']),
   impulse_error=abs(x['impulse_Ns']-ref['impulse_Ns'])/abs(ref['impulse_Ns']),
   minimum_gap_m=x['minimum_gap_m'],contact_work_J=x['contact_work_J'],
   energy_defect_J=x['max_total_energy_defect_J'],rollback_copy=x['forced_rejection_left_state_unchanged'])
 rows.append(row);(e/'fp64-midpoint-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(row,flush=True)
 if r.returncode!=0 or (row['force_error']<=.0025 and row['impulse_error']<=.0025):break
