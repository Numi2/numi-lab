from pathlib import Path
import gzip,hashlib,json,os,subprocess,shutil
R=Path.cwd();A=R/"artifacts/crow-v15-teacher-label-signal-fix-20261110";B=R/"build/crow-v15-teacher-label-signal-fix-20261110";P=R/"build/crow-v15-integrated-route-baseline-20261110"
base=json.loads((R/"artifacts/crow-v15-integrated-route-baseline-20261110/contract-bindings.json").read_text())["argv"]
for mode,shader in (("old",R/"build/crow-v15-route-teacher-earlier-turn-20261110/earlier.metallib"),("fixed",B/"labelfix.metallib")):
 cmd=list(base)
 for k,v in [("--metallib",str(shader)),("--birdflow-navigation-course","reserved-final-j"),("--seed","3193706597"),("--steps","2400"),("--policy-pack",str(P/"landing-diagnostic.policypack")),("--base-policy-pack",str(P/"route-diagnostic.policypack")),("--state-trace",str(B/(mode+".tsv")))]:
  cmd[cmd.index(k)+1]=v
 i=cmd.index("--crow-replay-pack");del cmd[i:i+2]
 cmd += ["--birdflow-journey-teacher","--birdflow-journey-student-authority","0","--rollout-pack",str(B/(mode+".rollout"))]
 env={**os.environ,"DYLD_LIBRARY_PATH":str(P),"CROW_CONTACT_TRACE":"1","PYTHONDONTWRITEBYTECODE":"1"}
 (A/(mode+"-bindings.json")).write_text(json.dumps({"argv":cmd},indent=2)+"\n")
 with (A/(mode+".stdout")).open("wb") as out,(A/(mode+".stderr.gz")).open("wb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz:
   p=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.PIPE);shutil.copyfileobj(p.stderr,gz);rc=p.wait()
 if rc:raise RuntimeError("run failed "+str(rc))
 print(mode,"complete",flush=True)
subprocess.run([str(B/"verify"),str(B/"old.rollout"),str(B/"fixed.rollout")],cwd=R,check=True)
for mode in ("old","fixed"):
 src=B/(mode+".rollout")
 with src.open("rb") as f,(A/(mode+".rollout.gz")).open("wb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz:shutil.copyfileobj(f,gz)
