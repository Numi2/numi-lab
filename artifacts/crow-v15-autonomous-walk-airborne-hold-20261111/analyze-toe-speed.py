from pathlib import Path
import gzip,collections,json
D=Path("artifacts/crow-v15-autonomous-walk-airborne-hold-20261111");out={}
for seed in (20261126,20261127):
 p=D/("autonomous-"+str(seed))/"run.stderr.gz";caps=collections.Counter();maximum=collections.defaultdict(float);coverage=collections.Counter()
 with gzip.open(p,"rt") as f:
  for l in f:
   if l.startswith("CROW_TOE_V_SUBSTEP "):
    x=l.split();t,k,e=map(int,x[1:4])
    if 1<=t<=600:
     part="settling" if t<=200 else "walk";v=max(abs(float(y)) for y in x[4:]);caps[part,e]+=int(v>=11.99);maximum[part,e]=max(maximum[part,e],v);coverage[e,t,k]+=1
 assert len(coverage)==8*600*16 and all(v==1 for v in coverage.values())
 out[str(seed)]={part:{str(e):{"cap_substeps":caps[part,e],"peak_rad_s":maximum[part,e]} for e in range(8)} for part in ("settling","walk")}
(D/"toe-speed-analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
