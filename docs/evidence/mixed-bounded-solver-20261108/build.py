from pathlib import Path
import json,subprocess
p=Path('/Users/numi/frictionless-mixed-rod-20261108');d=Path.cwd();old=json.loads((p/'split-build.json').read_text());result=[]
for cap in (2,64):
 for item in old:
  args=[x.replace(str(p/'split.metal'),str(d/f'candidate-{cap}.metal')).replace(str(p/'split.air'),str(d/f'candidate-{cap}.air')).replace(str(p/'split.metallib'),str(d/f'candidate-{cap}.metallib')) for x in item['command']]
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);result.append({'command':args,'exit':r.returncode,'output':r.stdout});(d/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(cap,r.returncode,r.stdout,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
