from pathlib import Path
import json,subprocess
old=json.loads(Path('failure-build.json').read_text());out=[]
for item in old:
 args=[x.replace('candidate-2e-4-reporting.metal','candidate-2e-4-cap128.metal').replace('candidate-2e-4-reporting.air','candidate-2e-4-cap128.air').replace('candidate-2e-4-reporting.metallib','candidate-2e-4-cap128.metallib').replace('candidate-2e-5-reporting.metal','candidate-2e-5-cap128.metal').replace('candidate-2e-5-reporting.air','candidate-2e-5-cap128.air').replace('candidate-2e-5-reporting.metallib','candidate-2e-5-cap128.metallib') for x in item['command']]
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);out.append({'command':args,'exit':r.returncode,'output':r.stdout});Path('cap-build.json').write_text(json.dumps(out,indent=2)+'\n');print(r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
