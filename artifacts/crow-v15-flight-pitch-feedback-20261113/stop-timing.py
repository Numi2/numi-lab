from pathlib import Path
import gzip,json,collections
A=Path("artifacts/crow-v15-flight-pitch-feedback-20261113");sources=[Path("artifacts/crow-v15-overlap-obstacle-free-20261113"),A];runs=["push-wing-20261126-rebound-clear","push-wing-20261126-rebound-pitch"];out=[]
for root,run in zip(sources,runs):
 p=root/run;meta=json.loads((p/"run.stdout").read_text())["first_episode_navigation"];analysis=json.loads((root/"analysis.json").read_text());recontact={x["lane"]:x["recontact"] for x in analysis["rows"]};stat=collections.defaultdict(lambda:{"first":None,"count":0,"before_recontact":0})
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
   x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
   if e<2 or t>meta["steps_by_environment"][e]:continue
   if max(abs(x["q"][7]),abs(x["q"][8]))>=.19999:
    z=stat[e];z["first"]=z["first"] or t;z["count"]+=1;z["before_recontact"]+=int(recontact[e] is not None and t<recontact[e])
 for e in range(2,8):out.append({"run":run,"lane":e,"first_foot_recontact":recontact[e],"first_sampled_sweep_stop":stat[e]["first"],"stop_samples":stat[e]["count"],"stop_samples_before_foot_recontact":stat[e]["before_recontact"]})
(A/"stop-timing.json").write_text(json.dumps({"rows":out,"scope":"Accepted control-step samples only, not a full substep stop observer."},indent=2)+"\n");print(json.dumps(out,indent=2))
