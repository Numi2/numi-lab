from pathlib import Path
import json,subprocess
old=json.loads(Path('threshold-build.json').read_text());out=[]
for item in old:
 args=[x.replace('candidate-2e-4.metal','candidate-2e-4-reporting.metal').replace('candidate-2e-4.air','candidate-2e-4-reporting.air').replace('candidate-2e-4.metallib','candidate-2e-4-reporting.metallib').replace('candidate-2e-5.metal','candidate-2e-5-reporting.metal').replace('candidate-2e-5.air','candidate-2e-5-reporting.air').replace('candidate-2e-5.metallib','candidate-2e-5-reporting.metallib') for x in item['command']]
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);out.append({'command':args,'exit':r.returncode,'output':r.stdout});Path('failure-build.json').write_text(json.dumps(out,indent=2)+'\n');print(r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
