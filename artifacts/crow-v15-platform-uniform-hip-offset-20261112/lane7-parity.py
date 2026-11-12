from pathlib import Path
import gzip, subprocess, hashlib, json
A=Path("artifacts/crow-v15-platform-uniform-hip-offset-20261112")
paths=[Path("artifacts/crow-v15-platform-hip-offset-screen-20261112/autonomous-20261126-posture"),A/"autonomous-20261126-posture"]
def digest(d):
 p=subprocess.Popen(["cat",*[str(x) for x in sorted(d.glob("run.stderr.gz.part[0-9][0-9][0-9]"))]],stdout=subprocess.PIPE)
 hs={k:hashlib.sha256() for k in ("state","contact")};counts={k:0 for k in hs};a={}
 with gzip.open(p.stdout,"rt") as z:
  for line in z:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"];key="state"
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    x=line.split(None,4);t,sub,e=map(int,x[1:4]);key="contact"
   else: continue
   if e==7 and t<=4800:
    hs[key].update(line.encode());counts[key]+=1
    if key=="state" and t in (1000,2000,3000,3800,4800): a[t]=x
 assert p.wait()==0
 return {k:{"count":counts[k],"sha256":hs[k].hexdigest()} for k in hs},a
a,aa=digest(paths[0]);b,bb=digest(paths[1]);r={"previous":a,"uniform":b,"identical":{k:a[k]==b[k] for k in a},"sample_endpoints_identical":{k:aa.get(k)==bb.get(k) for k in aa},"note":"Lane 7 raw accepted endpoint and contact records; exact hash equality required, numerical proximity is not exact replication."};(A/"lane7-parity.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
