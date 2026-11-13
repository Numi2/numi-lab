from pathlib import Path
import gzip,json,statistics,sys,math
P=Path("artifacts/crow-v15-mirrored-height-trim-20261113")/sys.argv[1];rows={6:[],7:[]}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
   if e in rows and 1250<=t<=1800:rows[e].append((t,x["q"][2],x["v"][2]))
out=[]
for e,a in rows.items():
 assert len(a)==551 and len({x[0] for x in a})==551;goal=.8 if e==6 else .55;h=[x[1] for x in a];v=[x[2] for x in a];out.append({"lane":e,"target_m":goal,"samples":len(a),"height_mean_m":statistics.mean(h),"height_min_m":min(h),"height_max_m":max(h),"height_mean_abs_error_m":statistics.mean(abs(x-goal) for x in h),"vertical_speed_rms_m_s":math.sqrt(statistics.mean(x*x for x in v)),"last_height_m":h[-1]})
(P/"tracking-analysis.json").write_text(json.dumps({"rows":out,"scope":"Posthoc fixed-window root-height tracking; body motion contains wingbeat oscillations. One used seed; not qualified altitude tracking."},indent=2)+"\n");print(json.dumps(out,indent=2))
