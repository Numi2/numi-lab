from pathlib import Path
import json,subprocess
old=json.loads(Path('/Users/numi/mixed-bounded-20261108/build.json').read_text())[:2];dst=Path.cwd();records=[]
for label in ('2e-4','2e-5'):
 for item in old:
  args=[x.replace('/Users/numi/mixed-bounded-20261108/candidate-2.metal',str(dst/f'candidate-{label}.metal')).replace('/Users/numi/mixed-bounded-20261108/candidate-2.air',str(dst/f'candidate-{label}.air')).replace('/Users/numi/mixed-bounded-20261108/candidate-2.metallib',str(dst/f'candidate-{label}.metallib')) for x in item['command']]
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);records.append({'command':args,'exit':r.returncode,'output':r.stdout});Path('threshold-build.json').write_text(json.dumps(records,indent=2)+'\n');print(label,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
