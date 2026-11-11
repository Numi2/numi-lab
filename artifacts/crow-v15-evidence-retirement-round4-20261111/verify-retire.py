from pathlib import Path
import gzip,hashlib,json,subprocess,tempfile,shutil
R=Path.cwd();I=Path("/tmp/crow-restore-check-20261110");A=R/"artifacts/crow-v15-evidence-retirement-round4-20261111"
subprocess.run(["git","-C",str(I),"fetch","origin","agent/american-crow-rl-20261107"],check=True)
commit=subprocess.check_output(["git","-C",str(I),"rev-parse","FETCH_HEAD"],text=True).strip()
def digest(path,unzip=False):
 h=hashlib.sha256();op=gzip.open if unzip else open
 with op(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def show(spec):return subprocess.check_output(["git","-C",str(I),"show",commit+":"+spec])
results=[]
def retire(raw,expected,source):
 raw=Path(raw);assert raw.exists() and digest(raw)==expected
 p=subprocess.run(["lsof","-t","--",str(raw)],capture_output=True,text=True);assert p.returncode in (0,1) and not p.stdout.strip()
 assert digest(raw)==expected;size=raw.stat().st_size;raw.unlink()
 results.append({"path":str(raw),"bytes":size,"sha256":expected,"remote_source":source,"retired":True})
 (A/"verification.json").write_text(json.dumps({"independently_fetched_commit":commit,"results":results},indent=2)+"\n")
with tempfile.TemporaryDirectory(prefix="crow-round4-") as td:
 td=Path(td);pfx="artifacts/crow-v15-perch-lower-approach-control-20261111/autonomous-20261126";m=json.loads(show(pfx+"/stderr-parts.json"));tmp=td/"lower.gz";h=hashlib.sha256()
 with tmp.open("xb") as o:
  for part in m["parts"]:
   data=show(pfx+"/"+part["name"]);assert len(data)==part["bytes"] and hashlib.sha256(data).hexdigest()==part["sha256"];o.write(data);h.update(data)
 assert h.hexdigest()==m["compressed_sha256"] and tmp.stat().st_size==m["bytes"]
 raw=R/pfx/"run.stderr.gz";assert digest(raw,True)==digest(tmp,True);retire(raw,m["compressed_sha256"],pfx+"/stderr-parts.json")
 pfx="artifacts/crow-v15-near-slalom-teacher-recovery-20261111/recovery-20261126";tmp=td/"teacher.gz"
 with tmp.open("wb") as o:subprocess.run(["git","-C",str(I),"show",commit+":"+pfx+"/rollout.bin.gz"],stdout=o,check=True)
 sums=show("artifacts/crow-v15-near-slalom-teacher-recovery-20261111/SHA256SUMS").decode();entry=next(x.split("  ",1)[0] for x in sums.splitlines() if x.endswith("recovery-20261126/rollout.bin.gz"));assert digest(tmp)==entry
 raw=R/"build/crow-v15-near-slalom-teacher-recovery-20261111/recovery-20261126-rollout.bin";retire(raw,digest(tmp,True),pfx+"/rollout.bin.gz")
 previous=json.loads((R/"artifacts/crow-v15-evidence-retirement-round2-20261111/result.json").read_text());row=next(x for x in previous["files_retired"] if x["raw_path"].endswith("crow-v15-student-state-teacher-recovery-20261111/recovery-20261126-rollout.bin"))
 retire(row["raw_path"],row["raw_sha256"],row["remote_object"])
print(json.dumps({"independently_fetched_commit":commit,"results":results},indent=2))
