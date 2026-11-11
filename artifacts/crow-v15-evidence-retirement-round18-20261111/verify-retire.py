from pathlib import Path
import gzip,hashlib,json,subprocess,tempfile
R=Path.cwd();I=Path("/tmp/crow-restore-check-20261110");A=R/"artifacts/crow-v15-evidence-retirement-round18-20261111";branch="agent/american-crow-rl-20261107"
subprocess.run(["git","-C",str(I),"fetch","origin",branch],check=True);c=subprocess.check_output(["git","-C",str(I),"rev-parse","FETCH_HEAD"],text=True).strip();out=[]
def digest(p,gz=False):
 h=hashlib.sha256()
 with (gzip.open if gz else open)(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def retire(local,restored,gz,ref):
 p=R/local;h=digest(p);assert h==digest(restored,gz);z=subprocess.run(["lsof","-t","--",str(p)],capture_output=True,text=True);assert z.returncode in (0,1) and not z.stdout.strip();assert h==digest(p);n=p.stat().st_size;p.unlink();out.append({"path":str(p),"bytes":n,"sha256":h,"verified_archive":ref,"open_handles":False,"retired":True});(A/"verification.json").write_text(json.dumps({"independently_fetched_commit":c,"results":out},indent=2)+"\n")
settings=[("crow-v15-landing-staged-recovery-20261111","autonomous-20261126","run.stderr.gz"),("crow-v15-low-speed-flight-screen-20261111","","flight.stderr.gz")]
with tempfile.TemporaryDirectory(prefix="crow-archive-restore-") as td:
 tmp=Path(td)
 for project,run,stderr in settings:
  item="artifacts/"+project+("/"+run if run else "");m=json.loads(subprocess.check_output(["git","-C",str(I),"show",c+":"+item+"/stderr-parts.json"]));restored=tmp/(project+".gz");h=hashlib.sha256()
  with restored.open("wb") as f:
   for x in m["parts"]:
    b=subprocess.check_output(["git","-C",str(I),"show",c+":"+item+"/"+x["name"]]);assert len(b)==x["bytes"] and hashlib.sha256(b).hexdigest()==x["sha256"];f.write(b);h.update(b)
  assert h.hexdigest()==m["compressed_sha256"] and restored.stat().st_size==m["bytes"];retire(item+"/"+stderr,restored,False,item+"/stderr-parts.json")
  for kind in ("replay.json","trace.tsv"):
   name=kind+".gz" if run else "flight-"+kind+".gz";remote=item+"/"+name;restored=tmp/(project+"-"+name);restored.write_bytes(subprocess.check_output(["git","-C",str(I),"show",c+":"+remote]));build="build/"+project+"/"+(run if run else "flight")+"-"+kind;retire(build,restored,True,remote)
print(json.dumps({"independently_fetched_commit":c,"results":out},indent=2))
