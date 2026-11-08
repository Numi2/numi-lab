import json,subprocess
from pathlib import Path
p=Path.cwd();old=json.loads(Path('build.json').read_text())[:2];out=[]
for item in old:
 args=[x.replace('candidate-2.metal','candidate-2-count-fixed.metal').replace('candidate-2.air','candidate-2-count-fixed.air').replace('candidate-2.metallib','candidate-2-count-fixed.metallib') for x in item['command']]
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);out.append({'command':args,'exit':r.returncode,'output':r.stdout});Path('count-build.json').write_text(json.dumps(out,indent=2)+'\n');print(r.returncode,r.stdout,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
