from pathlib import Path
import json,hashlib,time
N=Path(__file__).resolve().parent;p=N/"install-motion.plan.json";plan=json.loads(p.read_text());t=time.monotonic();records=[]
for domain in ["input_pins","role_input_pins"]:
 for raw,expected in plan[domain].items():
  path=Path(raw)
  with path.open("rb") as f: actual=hashlib.file_digest(f,"sha256").hexdigest()
  records.append({"domain":domain,"path":raw,"expected_sha256":expected,"actual_sha256":actual,"pass":actual==expected})
report={"status":"PASS" if all(r["pass"] for r in records) and records else "FAIL","plan_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"checked_files":len(records),"elapsed_s":time.monotonic()-t,"records":records,"qualification_scope":"supplements initial plan builder which did not recognize path-to-hash string dictionaries","physical_admission":False}
(N/"launch-input-pin-verification.json").write_text(json.dumps(report,indent=2)+"\n");print(report["status"],report["checked_files"],report["elapsed_s"])
assert report["status"]=="PASS"
