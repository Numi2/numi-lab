#!/usr/bin/env python3
from pathlib import Path
import csv,json,hashlib,sys
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
BASE=Q/"solver/fresh-damped-cgls-complete-coupon-521/output"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keyed(p):
 with open(p,newline="") as f: rr=list(csv.DictReader(f))
 out={}
 for r in rr:
  k=(r["step"],r["environment"],r["arm"])
  if k in out:raise ValueError("duplicate observation key "+repr(k))
  out[k]=r
 return out
def prefix(d,e):
 xs=sorted(int(r["step"]) for r in d.values() if r["environment"]==str(e) and r["arm"]=="bent" and r["step_accepted"]=="1");n=0
 for x in xs:
  if x!=n:break
  n+=1
 return n
def main(newout,dest):
 old=keyed(BASE/"observations.csv");new=keyed(Path(newout)/"observations.csv");common=sorted(set(old)&set(new));diff=[]
 for k in common:
  fs=[f for f in old[k] if old[k][f]!=new[k].get(f)]
  if fs and len(diff)<100:diff.append({"key":k,"fields":fs})
 eq=sum(not any(old[k][f]!=new[k].get(f) for f in old[k]) for k in common)
 geo=[]
 for p in sorted(BASE.glob("accepted_step_*.obj")):
  q=Path(newout)/p.name
  if q.exists():geo.append({"name":p.name,"source198_sha256":sha(p),"source201_sha256":sha(q),"identical":sha(p)==sha(q)})
 envs=sorted({int(r["environment"]) for r in old.values()}|{int(r["environment"]) for r in new.values()})
 out={"schema":"numi.cardboard.native535-prefix-comparison.v1","comparison_only":True,"common_observation_rows":len(common),"old_rows":len(old),"new_rows":len(new),"identical_common_rows":eq,"different_common_rows":len(common)-eq,"difference_examples":diff,"per_environment_accepted_bent_prefix_controls":{str(e):{"source198_521":prefix(old,e),"source203_539":prefix(new,e)} for e in envs},"common_accepted_geometry":geo,"limitations":["Descriptive only; not an acceptance gate.","Do not require environment1 bit parity because source203 changes the restart dispatch behavior.","An accepted prefix does not establish completion or physical qualification."]}
 Path(dest).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"common_rows":len(common),"different":len(common)-eq,"prefix":out["per_environment_accepted_bent_prefix_controls"]}))
if __name__=="__main__":
 if len(sys.argv)!=3:raise SystemExit("usage: compare_prefixes.py NEW_OUTPUT REPORT")
 main(sys.argv[1],sys.argv[2])
