from pathlib import Path
import hashlib,json,os,subprocess,sys
A=Path(__file__).resolve().parent;R=A.parent;p=R/"crow-v15-reset-corrected-leg-ablation-20261114/hold-20261126-rebound-fixed500/run.stderr.gz";v=json.loads((R/"crow-v15-heldleg-verification-20261114/verification.json").read_text());m=json.loads((p.parent/"raw-manifest.json").read_text());assert v["gzip_eof"] and v["sha256"]==m["sha256"] and v["bytes"]==m["bytes"];assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_size==v["bytes"];h=hashlib.sha256()
with p.open("rb") as f:
 for b in iter(lambda:f.read(1<<20),b""):h.update(b)
assert h.hexdigest()==v["sha256"];r=subprocess.run(["/usr/sbin/lsof","-t","--",str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout;s2=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns);out={"path":str(p),"inode":s.st_ino,"bytes":s.st_size,"mtime_ns":s.st_mtime_ns,"sha256":h.hexdigest(),"lsof_clear":True,"retired":False}
if "--retire" in sys.argv:
 pre=json.loads((A/"local-preflight.json").read_text());assert pre==out;os.unlink(p);out["retired"]=True
print(json.dumps(out,indent=2))
