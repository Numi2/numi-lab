import itertools,json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-paired-step-20261108")
with (p/"holdout-corrected.jsonl").open("x") as out:
 for offset,kind,repeat in itertools.product(("0.4","1.25","100"),("fixed","moving","internal"),range(2)):
  cmd=[str(p/"probe-holdout"),str(p/"candidate.metallib"),"6","x",offset,kind]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
  out.write(json.dumps(dict(n=6,axis="x",offset=offset,kind=kind,repeat=repeat,command=cmd,exit=r.returncode,stdout=r.stdout,stderr=r.stderr))+"\n");out.flush()
  if r.returncode or r.stderr:print("ERROR",offset,kind,repeat,r.returncode,r.stderr,flush=True)
print("completed 18",flush=True)
