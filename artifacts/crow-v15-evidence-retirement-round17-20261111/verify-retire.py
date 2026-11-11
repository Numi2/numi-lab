from pathlib import Path
import gzip,hashlib,json,subprocess,tempfile
R=Path.cwd(); I=Path("/tmp/crow-restore-check-20261110"); A=R/"artifacts/crow-v15-evidence-retirement-round17-20261111"; branch="agent/american-crow-rl-20261107"
subprocess.run(["git","-C",str(I),"fetch","origin",branch],check=True)
c=subprocess.check_output(["git","-C",str(I),"rev-parse","FETCH_HEAD"],text=True).strip(); item="artifacts/crow-v15-landing-tip-recovery-20261111/autonomous-20261126"; results=[]
def digest(path,compressed=False):
 h=hashlib.sha256(); op=gzip.open if compressed else open
 with op(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def retire(local, restored, decompress, description):
 source=R/local; orig=digest(source); expanded=digest(restored,decompress); assert orig==expanded,(source,orig,expanded)
 r=subprocess.run(["lsof","-t","--",str(source)],capture_output=True,text=True); assert r.returncode in (0,1) and not r.stdout.strip(); assert digest(source)==orig
 size=source.stat().st_size; source.unlink(); results.append({"path":str(source),"bytes":size,"sha256":orig,"verified_archive":description,"open_handles":False,"retired":True});(A/"verification.json").write_text(json.dumps({"independently_fetched_commit":c,"results":results},indent=2)+"\n")
with tempfile.TemporaryDirectory(prefix="crow-evidence-restore-") as td:
 tmp=Path(td); m=json.loads(subprocess.check_output(["git","-C",str(I),"show",c+":"+item+"/stderr-parts.json"])); raw=tmp/"stderr.gz"; h=hashlib.sha256()
 with raw.open("wb") as out:
  for p in m["parts"]:
   b=subprocess.check_output(["git","-C",str(I),"show",c+":"+item+"/"+p["name"]]); assert len(b)==p["bytes"] and hashlib.sha256(b).hexdigest()==p["sha256"]; out.write(b);h.update(b)
 assert h.hexdigest()==m["compressed_sha256"] and raw.stat().st_size==m["bytes"]; retire(item+"/run.stderr.gz",raw,False,item+"/stderr-parts.json")
 for kind in ("replay.json","trace.tsv"):
  gz=tmp/(kind+".gz"); gz.write_bytes(subprocess.check_output(["git","-C",str(I),"show",c+":"+item+"/"+kind+".gz"])); retire("build/crow-v15-landing-tip-recovery-20261111/autonomous-20261126-"+kind,gz,True,item+"/"+kind+".gz")
print(json.dumps({"independently_fetched_commit":c,"results":results},indent=2))
