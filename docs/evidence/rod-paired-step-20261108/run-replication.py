import itertools,json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-paired-step-20261108")
with (p/"replication.jsonl").open("x") as out:
 for n,axis,offset,kind,repeat in itertools.product((5,8),("x","oblique3"),("0","0.25"),("fixed","moving","internal"),range(2)):
  cmd=[str(p/"probe-paired"),str(p/"candidate.metallib"),str(n),axis,offset,kind]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
  out.write(json.dumps(dict(n=n,axis=axis,offset=offset,kind=kind,repeat=repeat,command=cmd,exit=r.returncode,stdout=r.stdout,stderr=r.stderr))+"\n");out.flush()
  if r.returncode or r.stderr:print("ERROR",n,axis,offset,kind,repeat,r.returncode,r.stderr,flush=True)
print("completed 48",flush=True)
