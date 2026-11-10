from pathlib import Path
import gzip,hashlib,json,subprocess
R=Path.cwd(); C=Path("/tmp/crow-restore-check-20261110");rev="730e28901";branch="agent/american-crow-rl-20261107"
subprocess.run(["git","-C",str(C),"fetch","--quiet","origin",branch],check=True)
actual=subprocess.check_output(["git","-C",str(C),"rev-parse","FETCH_HEAD"],text=True).strip()
if not actual.startswith(rev):raise SystemExit("remote tip differs from expected: "+actual)
def digest(f):
 h=hashlib.sha256()
 for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
mapping=[]
for n in ("i","j"):
 for ext in ("json","tsv"):
  name="development-reserved-final-"+n+"."+ext;mapping.append(("crow-v15-guide-foot-alignment-20261110",name))
for n in ("off","labels"):
 for ext in ("json","tsv"):mapping.append(("crow-v15-teacher-student-causality-20261110",n+"."+ext))
mapping.append(("crow-v15-sequenced-toe-release-20261110","results.txt"));retired=[]
for d,n in mapping:
 raw=R/"build"/d/n;arc="artifacts/"+d+"/"+n+".gz"
 if not raw.is_file():raise SystemExit("missing raw "+str(raw))
 proc=subprocess.Popen(["git","-C",str(C),"show",actual+":"+arc],stdout=subprocess.PIPE)
 with gzip.GzipFile(fileobj=proc.stdout,mode="rb") as f:restored=digest(f)
 if proc.wait()!=0:raise SystemExit("restore failed "+arc)
 with raw.open("rb") as f:original=digest(f)
 if restored!=original:raise SystemExit("mismatch "+arc)
 retired.append({"raw":str(raw.relative_to(R)),"archive":arc,"sha256":original,"bytes":raw.stat().st_size})
for x in retired:(R/x["raw"]).unlink()
(Path("artifacts/crow-v15-release-route-archive-check-20261110/retired.json")).write_text(json.dumps({"remote_commit":actual,"retired":retired,"bytes":sum(x["bytes"] for x in retired)},indent=2)+"\n")
print("restored, matched and retired",len(retired),"raw duplicates",sum(x["bytes"] for x in retired),"bytes")
