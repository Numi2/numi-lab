from pathlib import Path
import collections,gzip,hashlib,json,subprocess
D=Path("artifacts/crow-v15-landing-target-observer-20261111");P=D/"autonomous-20261126";R=Path("artifacts/crow-v15-landing-staged-long-horizon-20261111/autonomous-20261126");ends=[min(a,b) for a,b in zip(json.loads((P/"run.stdout").read_text())["first_episode_navigation"]["steps_by_environment"],json.loads((R/"run.stdout").read_text())["first_episode_navigation"]["steps_by_environment"])];fields=("q","v","policy_action","contacts")
def read(path):
 hs={(e,k):hashlib.sha256() for e in range(8) for k in ("endpoint","phase","substep")};cnt=collections.Counter();targets=collections.Counter();proc=None
 if (path/"run.stderr.gz").exists():stream=gzip.open(path/"run.stderr.gz","rt")
 else:
  m=json.loads((path/"stderr-parts.json").read_text());proc=subprocess.Popen(["cat"]+[str(path/z["name"]) for z in m["parts"]],stdout=subprocess.PIPE);stream=gzip.open(proc.stdout,"rt")
 with stream as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"];kind="endpoint";payload=json.dumps([t]+[x[k] for k in fields],sort_keys=True,separators=(",",":"))
   elif line.startswith("CROW_PHASE_SAMPLE "):
    z=line.split();t,e=map(int,z[1:3]);kind="phase";payload=" ".join(z[1:])
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);kind="substep";payload=" ".join(z[1:])
   elif line.startswith("CROW_DRIVE_TARGET "):
    z=line.split();t,e=map(int,z[1:3]);
    if 3200<=t<=min(4000,ends[e]) and len(z[3:])==40:targets[e]+=1
    continue
   else:continue
   if t<=ends[e]:hs[e,kind].update((payload+"\n").encode());cnt[e,kind]+=1
 if proc is not None and proc.wait()!=0:raise RuntimeError("archive read failed")
 return hs,cnt,targets
a=read(P);b=read(R);rows=[]
for e,end in enumerate(ends):
 z={"lane":e,"common_first_episode_end":end}
 for kind,expected in (("endpoint",end),("phase",end),("substep",16*end)):
  z[kind+"_count_observed"]=a[1][e,kind];z[kind+"_count_reference"]=b[1][e,kind];z[kind+"_hash_equal"]=a[0][e,kind].hexdigest()==b[0][e,kind].hexdigest();z[kind+"_expected_count"]=expected
 z["target_samples"]=a[2][e];z["target_expected"]=max(0,min(end,4000)-3199);rows.append(z)
out={"rows":rows,"scope":"Hashes compare logged q/v, actor outputs, reported endpoint contacts, phase records, and aggregate contact substeps in first episodes; they do not prove parity of unlogged internals. Target-buffer timing and mapping need separate verification. Additional self-contact messages are intentionally excluded from parity hashes."};(D/"reference-comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
