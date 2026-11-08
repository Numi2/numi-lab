import json,subprocess
from pathlib import Path
p=Path('/Users/numi/mixed-bounded-20261108');old=json.loads((p/'build.json').read_text())[:2];out=[]
for item in old:
 args=[x.replace('candidate-2.metal','candidate-trace.metal').replace('candidate-2.air','candidate-trace.air').replace('candidate-2.metallib','candidate-trace.metallib') for x in item['command']]
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);out.append({'command':args,'exit':r.returncode,'output':r.stdout});(p/'trace-build.json').write_text(json.dumps(out,indent=2)+'\n');print(r.returncode,r.stdout,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
