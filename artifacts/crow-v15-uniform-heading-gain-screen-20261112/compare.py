from pathlib import Path
import gzip,hashlib,json
A=Path("artifacts/crow-v15-uniform-heading-gain-screen-20261112");B=Path("artifacts/crow-v15-flight-diagnostic-state-isolation-20261112");name="candidate-20261126-rebound-hip"
def read(root):
 out={}
 with gzip.open(root/name/"run.stderr.gz","rb") as f:
  for line in f:
   if line.startswith(b"CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(b" ",1)[1]);out[(x["lane"],x["step"])]=hashlib.sha256(line).hexdigest()
 return out
base=read(B);candidate=read(A)
bmeta=json.loads((B/name/"run.stdout").read_text());cmeta=json.loads((A/name/"run.stdout").read_text())
bends=bmeta["first_episode_navigation"]["steps_by_environment"];cends=cmeta["first_episode_navigation"]["steps_by_environment"]
rows=[]
for e in range(8):
 end=min(bends[e],cends[e]); keys=[(e,t) for t in range(2,end+2)]
 missing=[t for _,t in keys if (e,t) not in base or (e,t) not in candidate]
 different=[t for _,t in keys if (e,t) in base and (e,t) in candidate and base[e,t]!=candidate[e,t]]
 rows.append({"lane":e,"base_steps":bends[e],"candidate_steps":cends[e],"missing_steps":missing,"first_different_trace_submission":min(different) if different else None})
with gzip.open(B/name/"replay.json.gz","rt") as f: br=json.load(f)["payload"]["frames"]
with gzip.open(A/name/"replay.json.gz","rt") as f: cr=json.load(f)["payload"]["frames"]
assert len(br)==len(cr)
reward_differences=[{"frame":i,"base":x["reward"],"candidate":y["reward"]} for i,(x,y) in enumerate(zip(br,cr)) if x["reward"]!=y["reward"]]
out={"same_first_episode_navigation":bmeta["first_episode_navigation"]==cmeta["first_episode_navigation"],"replay_environment":5,"replay_reward_difference_count":len(reward_differences),"first_replay_reward_difference":next(iter(reward_differences),None),"rows":rows,"note":"State hashes include q, v, accepted policy actions and contact details; reward comparison covers only replay environment 5, not a full reward decomposition. Trace submission includes a pre-reset warmup."}
(A/"parity.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
