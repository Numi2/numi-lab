from pathlib import Path
import gzip,json,collections
A=Path("artifacts/crow-v15-flight-fold-self-contact-observer-20261111")
refs={"reference":Path("build/crow-v15-contact-gated-flight-toes-20261111/flight-replay.json"),"fold":Path("build/crow-v15-flight-leg-fold-control-20261111/flight-replay.json")}
out={}
for mode,ref in refs.items():
 with gzip.open(A/mode/"replay.json.gz","rt") as f:current=json.load(f)["payload"]
 with ref.open() as f:original=json.load(f)["payload"]
 assert current["environment"]==original["environment"]==1
 frames=current["frames"];old=original["frames"];assert len(frames)==600 and len(old)>=600
 dif=[i+1 for i,(x,y) in enumerate(zip(frames,old[:600])) if any(x[k]!=y[k] for k in ("q","v","actor_observation","accepted_actions"))]
 counts=collections.Counter();maxima={};examples={};substeps=collections.defaultdict(set)
 with gzip.open(A/mode/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SELF_CONTACT "):
    z=line.split();t,k,e,a,b=map(int,z[1:6]);j=float(z[6]);sep=float(z[7]);key=(e,a,b)
    counts[key]+=1;substeps[key].add((t,k));maxima[key]=max(j,maxima.get(key,0));examples.setdefault(key,{"step":t,"substep":k,"separation_m":sep})
 out[mode]={"replay_lane":1,"replay_compared_frames":len(frames),"replay_first_difference":dif[:1],"replay_different_frames":len(dif),"self_contacts":[{"lane":e,"bodyA":a,"bodyB":b,"contact_samples":n,"distinct_step_substeps":len(substeps[e,a,b]),"bodyA_name":original["body_names"][a],"bodyB_name":original["body_names"][b],"maximum_normal_impulse":maxima[e,a,b],"first":examples[e,a,b]} for (e,a,b),n in sorted(counts.items())]}
out["scope"]="Diagnostic self-contact trace only at substep endpoints 380–600, positive normal impulses above 1e-7. Reference and fold use matched bounded-command obstacle-free runs. Full accepted q/v/action/observation parity is checked only for replay lane 1, not all lanes. Self-contact presence cannot alone establish causation."
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
