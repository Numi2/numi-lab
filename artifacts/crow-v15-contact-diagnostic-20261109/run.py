from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-contact-diagnostic-20261109";B=R/"build/crow-v15-contact-diagnostic-20261109";C=R/"build/crow-v15-toe-support-20261109";old=json.loads((R/"artifacts/crow-v15-toe-support-20261109/full-reserved-final-j-active-bindings.json").read_text());cmd=old["argv"]
for k,v in [("--state-trace",str(B/"diagnostic.tsv")),("--crow-replay-pack",str(B/"diagnostic.json"))]:cmd[cmd.index(k)+1]=v
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();old.update({"argv":cmd,"library_sha256":h(B/"libmetalrobo.dylib"),"metallib_sha256":h(C/"active.metallib"),"diagnostic":"shared task-state buffer read after each completed command; no writes"});(A/"bindings.json").write_text(json.dumps(old,indent=2)+"\n");env={**os.environ,"DYLD_LIBRARY_PATH":str(B),"CROW_TASK_STATE_DIAGNOSTIC":"1","CROW_CONTACT_TRACE":"1","CROW_SUPPORT_TRACE":"1","CROW_CLEARANCE_TRACE":"1","PYTHONDONTWRITEBYTECODE":"1"};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600);(A/"stdout").write_text(p.stdout)
for suffix,data in [("stderr",p.stderr.encode()),("replay.json",(B/"diagnostic.json").read_bytes()),("trace.tsv",(B/"diagnostic.tsv").read_bytes())]:
 with (A/(suffix+".gz")).open("wb") as f:
  with gzip.GzipFile(fileobj=f,mode="wb",mtime=0) as z:z.write(data)
print("exit",p.returncode);raise SystemExit(p.returncode)
