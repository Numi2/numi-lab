from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys,threading
R=Path.cwd();A=R/"artifacts/crow-v15-slow-flight-fold-control-20261111";B=R/"build/crow-v15-slow-flight-fold-control-20261111"
mode=sys.argv[1];assert mode == "slow"
D=A/mode;D.mkdir(exist_ok=False)
cmd=json.loads((R/"artifacts/crow-v15-flight-leg-fold-control-20261111/rebound-command.json").read_text())
shader=R/"build/crow-v15-slow-flight-fold-control-20261111/tuck-flight.metallib"
for k,v in [("--steps","600"),("--metallib",str(shader)),("--crow-replay-pack",str(B/(mode+"-replay.json"))),("--state-trace",str(B/(mode+"-trace.tsv")))]:cmd[cmd.index(k)+1]=v
cmd[0]=str(R/"build/crow-v15-replay-writer-roundtrip-20261111/metalrobo_task_rollout")
(D/"command.json").write_text(json.dumps(cmd,indent=2)+"\n")
env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-flight-fold-self-contact-observer-20261111"),"CROW_EARLY_FLIGHT_TRACE":"1","CROW_CONTACT_TRACE":"1","CROW_SUBSTEP_TRACE":"1","PYTHONDONTWRITEBYTECODE":"1"}
p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path,compressed):
 with path.open("wb") as raw:
  if compressed:
   with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as out:
    for b in iter(lambda:src.read(1<<16),b""):out.write(b)
  else:
   for b in iter(lambda:src.read(1<<16),b""):raw.write(b)
threads=[threading.Thread(target=pump,args=(p.stdout,D/"run.stdout",False)),threading.Thread(target=pump,args=(p.stderr,D/"run.stderr.gz",True))]
for t in threads:t.start()
rc=p.wait()
for t in threads:t.join()
for kind in ("replay.json","trace.tsv"):
 src=B/(mode+"-"+kind)
 with src.open("rb") as f,(D/(kind+".gz")).open("wb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as z:
   for b in iter(lambda:f.read(1<<20),b""):z.write(b)
(D/"run-result.json").write_text(json.dumps({"exit_code":rc},indent=2)+"\n")
if rc:raise SystemExit(rc)
print(mode,"exit",rc,flush=True)
