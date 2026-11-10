from pathlib import Path
import gzip,hashlib,json,os,subprocess,shutil
R=Path.cwd();A=R/"artifacts/crow-v15-route-cross-behind-20261110";B=R/"build/crow-v15-integrated-route-baseline-20261110"
base=json.loads((R/"artifacts/crow-v15-integrated-route-baseline-20261110/contract-bindings.json").read_text())["argv"]
for mode in ("recovery",):
 for course,seed in (("reserved-final-i",4038889268),("reserved-final-j",3193706597)):
  cmd=list(base)
  for k,v in [("--metallib",str(R/"build/crow-v15-route-cross-behind-20261110/crossbehind.metallib")),("--birdflow-navigation-course",course),("--seed",str(seed)),("--steps","3000"),("--envs","8"),("--policy-pack",str(B/"landing-diagnostic.policypack")),("--base-policy-pack",str(B/"route-diagnostic.policypack")),("--state-trace",str(R/"build/crow-v15-route-cross-behind-20261110"/(course+".tsv")))]:
   cmd[cmd.index(k)+1]=v
  i=cmd.index("--crow-replay-pack");del cmd[i:i+2]
  if mode=="recovery":cmd += ["--birdflow-journey-teacher","--birdflow-journey-student-authority","0"]
  env={**os.environ,"DYLD_LIBRARY_PATH":str(B),"CROW_CONTACT_TRACE":"1","CROW_CLEARANCE_TRACE":"1","PYTHONDONTWRITEBYTECODE":"1"}
  (A/(mode+"-"+course+"-bindings.json")).write_text(json.dumps({"argv":cmd,"note":mode+"; first episode only; landing failures retained"},indent=2)+"\n")
  with (A/(mode+"-"+course+".stdout")).open("wb") as out,(A/(mode+"-"+course+".stderr.gz")).open("wb") as raw:
   with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz:
    p=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.PIPE);shutil.copyfileobj(p.stderr,gz);rc=p.wait()
  print(mode,course,"exit",rc,flush=True)
  if rc:raise SystemExit(rc)
