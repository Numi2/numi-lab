from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64,hashlib,json,subprocess
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D");T=D/"publication-202-stage-001"
repo="repos/Numi2/numi-lab";branch="evidence/human-hand-p202-20261027"
def sha(b):return hashlib.sha256(b).hexdigest()
def api(method,path,data=None):
 args=["gh","api","--method",method,path]
 if data is not None:args += ["--input","-"]
 p=subprocess.run(args,input=json.dumps(data,separators=(",",":")).encode() if data is not None else None,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
 if p.returncode:raise RuntimeError(path+": "+p.stderr.decode(errors="replace")[-1201:])
 return json.loads(p.stdout)
head=api("GET",repo+"/git/ref/heads/main")["object"]["sha"]
check=subprocess.run(["gh","api",repo+"/git/ref/heads/"+branch],capture_output=True,text=True,timeout=60)
if check.returncode==0:raise RuntimeError("branch exists; refusing overwrite")
if "404" not in check.stderr:raise RuntimeError("cannot confirm branch absent: "+check.stderr[-1201:])
manifest=json.loads((T/"publication-manifest.json").read_text())
files=sorted(p for p in T.rglob("*") if p.is_file())
lookup={p.relative_to(T).as_posix():p for p in files}
assert set(lookup)=={m["path"] for m in manifest["members"]}|{"publication-manifest.json"}
for entry in manifest["members"]:
 data=lookup[entry["path"]].read_bytes()
 assert len(data)==entry["bytes"] and sha(data)==entry["sha256"],entry["path"]
def upload(p):
 data=p.read_bytes();gitsha=hashlib.sha1(("blob "+str(len(data))+"\0").encode()+data).hexdigest()
 result=api("POST",repo+"/git/blobs",{"content":base64.b64encode(data).decode(),"encoding":"base64"})
 assert result["sha"]==gitsha
 return {"path":"publication-202/"+p.relative_to(T).as_posix(),"mode":"100644","type":"blob","sha":gitsha}
with ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(upload,files))
print("uploaded",len(entries),"blobs",flush=True)
tree=api("POST",repo+"/git/trees",{"tree":entries})["sha"]
commit=api("POST",repo+"/git/commits",{"message":"Archive human hand connected topology and layer diagnostics","tree":tree,"parents":[]})["sha"]
assert api("GET",repo+"/git/ref/heads/main")["object"]["sha"]==head
api("POST",repo+"/git/refs",{"ref":"refs/heads/"+branch,"sha":commit})
assert api("GET",repo+"/git/ref/heads/"+branch)["object"]["sha"]==commit
assert api("GET",repo+"/git/ref/heads/main")["object"]["sha"]==head
print("published evidence-only archive",branch,commit,flush=True)
listing=api("GET",repo+"/git/trees/"+tree+"?recursive=1")
assert not listing.get("truncated")
remote={v["path"].removeprefix("publication-202/"):v["sha"] for v in listing["tree"] if v["type"]=="blob"}
assert set(remote)==set(lookup)
def verify(item):
 path,blob=item;r=api("GET",repo+"/git/blobs/"+blob);assert r["encoding"]=="base64";data=base64.b64decode(r["content"]);expected=lookup[path].read_bytes()
 assert sha(data)==sha(expected) and len(data)==len(expected)
 return {"path":path,"sha256":sha(data),"bytes":len(data),"git_blob_sha":blob}
with ThreadPoolExecutor(max_workers=4) as pool:checked=list(pool.map(verify,remote.items()))
receipt={"repository":"Numi2/numi-lab","branch":branch,"reference_main":head,"commit":commit,"tree":tree,"status":"evidence-only archive; all files independently downloaded and sha256 verified","files":checked,"file_count":len(checked),"main_unchanged":True,"accepted_physical_steps":0,"geometry_admitted":False}
(D/"publication-202-remote-verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
print("independently verified",len(checked),"files",flush=True)
