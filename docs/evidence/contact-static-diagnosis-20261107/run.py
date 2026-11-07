import json,pathlib,subprocess,time
e=pathlib.Path('/Users/numi/numi-lab-contact-static-evidence-20261107')
exe='/Users/numi/numi-lab-contact-area-evidence-20261107/fem-plane-static-affine'
rows=[]
for n in (2,4,8,16,32):
 for rule in ('nodal','gaussian'):
  t=time.monotonic();r=subprocess.run([exe,str(n),rule],capture_output=True,text=True)
  (e/f'n{n}-{rule}.json').write_text(r.stdout)
  (e/f'n{n}-{rule}.stderr').write_text(r.stderr)
  row={'n':n,'rule':rule,'exit':r.returncode,'wall_seconds':time.monotonic()-t}
  rows.append(row);(e/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(row,flush=True)
  if r.returncode:raise SystemExit(1)
