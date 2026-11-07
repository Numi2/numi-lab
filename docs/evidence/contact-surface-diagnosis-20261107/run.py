import json,pathlib,subprocess,time
e=pathlib.Path('/Users/numi/numi-lab-contact-surface-evidence-20261107')
exe='/Users/numi/numi-lab-contact-area-evidence-20261107/fem-plane-surface-diagnostic'
cases=[(2,5e-7,'nodal'),(4,5e-7,'nodal'),(8,5e-7,'nodal'),
       (16,5e-7,'nodal'),(8,1e-6,'gaussian'),(8,5e-7,'gaussian')]
rows=[]
for n,h,mode in cases:
 t=time.monotonic();r=subprocess.run([exe,str(n),str(h),'.01',mode],capture_output=True,text=True)
 name=f'n{n}-h{h}-{mode}'
 (e/(name+'.json')).write_text(r.stdout);(e/(name+'.stderr')).write_text(r.stderr)
 row={'n':n,'h':h,'quadrature':mode,'wall_seconds':time.monotonic()-t,'exit':r.returncode}
 rows.append(row);(e/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(row,flush=True)
 if r.returncode:raise SystemExit(1)
