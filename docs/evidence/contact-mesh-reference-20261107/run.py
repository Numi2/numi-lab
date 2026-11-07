import json,pathlib,subprocess,time
e=pathlib.Path('/Users/numi/numi-lab-contact-mesh-evidence-20261107')
exe='/Users/numi/numi-lab-contact-area-evidence-20261107/fem-plane-fp64-reference'
rows=[]
for n in (4,8):
 for h in (1e-6,5e-7):
  t=time.monotonic();r=subprocess.run([exe,str(n),str(h),'.01'],capture_output=True,text=True)
  s=f'n{n}-h{h}'
  (e/(s+'.json')).write_text(r.stdout);(e/(s+'.stderr')).write_text(r.stderr)
  rows.append(dict(n=n,h=h,wall_seconds=time.monotonic()-t,exit=r.returncode))
  (e/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(rows[-1],flush=True)
  if r.returncode:raise SystemExit(1)
