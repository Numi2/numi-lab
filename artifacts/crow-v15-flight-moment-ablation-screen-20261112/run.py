from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys,threading
R=Path.cwd();D=R/"artifacts/crow-v15-flight-moment-ablation-screen-20261112";B=R/"build/crow-v15-flight-moment-ablation-screen-20261112"; reference=R/"artifacts/crow-v15-journey-bounded-flight-screen-20261111";name=sys.argv[1];assert name in ("preflight","flight"); steps=2 if name=="preflight" else 1600
cmd=json.loads((reference/"rebound-command.json").read_text())
for k,v in [("--metallib",str(R/"build/crow-v15-flight-moment-ablation-screen-20261112/feedback.metallib")),("--envs","12"),("--steps",str(steps)),("--seed","20261126"),("--state-trace",str(B/(name+"-trace.tsv"))),("--crow-replay-pack",str(B/(name+"-replay.json")))]:cmd[cmd.index(k)+1]=v
(D/(name+"-command.json")).write_text(json.dumps(cmd,indent=2)+"\n")
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();L=R/"build/crow-v15-hip-roll-obstacle-free-control-20261111";(D/(name+"-inputs.json")).write_text(json.dumps({"host_sha256":h(L/"libmetalrobo.dylib"),"metallib_sha256":h(R/"build/crow-v15-flight-moment-ablation-screen-20261112/feedback.metallib"),"frozen_flight_sha256":h(R/"artifacts/crow-v15-flight-feedback-freeze-20261110/CrowFlightControllerResearchV2.metal"),"policy_sha256":h(Path(cmd[cmd.index("--policy-pack")+1])),"base_policy_sha256":h(Path(cmd[cmd.index("--base-policy-pack")+1]))},indent=2)+"\n")
env={**os.environ,"DYLD_LIBRARY_PATH":str(L),"CROW_EARLY_FLIGHT_TRACE":"1","CROW_CONTACT_TRACE":"1","CROW_SUBSTEP_TRACE":"1","PYTHONDONTWRITEBYTECODE":"1"};p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path,compressed):
 with path.open("wb") as f:
  if compressed:
   with gzip.GzipFile(fileobj=f,mode="wb",mtime=0) as out:
    for b in iter(lambda:src.read(1<<16),b""):out.write(b)
  else:
   for b in iter(lambda:src.read(1<<16),b""):f.write(b)
ts=[threading.Thread(target=pump,args=(p.stdout,D/(name+".stdout"),False)),threading.Thread(target=pump,args=(p.stderr,D/(name+".stderr.gz"),True))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
for kind in ("trace.tsv","replay.json"):
 src=B/(name+"-"+kind)
 if src.exists():
  with src.open("rb") as i,(D/(name+"-"+kind+".gz")).open("wb") as raw:
   with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as out:
    for b in iter(lambda:i.read(1<<16),b""):out.write(b)
(D/(name+"-result.json")).write_text(json.dumps({"exit_code":rc},indent=2)+"\n");print(name,"exit",rc,flush=True)
if rc:sys.exit(rc)
