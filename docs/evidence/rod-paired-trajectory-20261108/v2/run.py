import itertools,json,subprocess,time
from pathlib import Path
p=Path(__file__).parent
binary=p.parent/"trajectory-v2";lib=Path("/Users/numi/rod-paired-step-20261108/candidate.metallib")
with (p/"manifest.jsonl").open("x") as manifest:
 for offset,kind,run in itertools.product(("0.25","100"),("fixed","moving","internal"),range(2)):
  stem=f"o{offset}-{kind}-r{run}";cmd=[str(binary),str(lib),offset,kind,str(run)]
  start=time.time()
  with (p/(stem+".jsonl")).open("x") as out,(p/(stem+".stderr")).open("x") as err:
   try:r=subprocess.run(cmd,stdout=out,stderr=err,timeout=300);exitcode=r.returncode;error=""
   except subprocess.TimeoutExpired as e:exitcode=None;error=str(e)
  manifest.write(json.dumps({"case":stem,"command":cmd,"exit":exitcode,"error":error,"wall_s":time.time()-start})+"\n");manifest.flush()
  print(stem,"exit",exitcode,"wall",round(time.time()-start,2),flush=True)
print("completed 12",flush=True)
