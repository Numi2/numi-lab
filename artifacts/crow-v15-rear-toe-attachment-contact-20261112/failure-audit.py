from pathlib import Path
import gzip,json,math,collections
p=Path("artifacts/crow-v15-rear-toe-attachment-contact-20261112");name="candidate-20261127";D=p/name;ep=json.loads((D/"run.stdout").read_text())["first_episode_navigation"];term=ep["steps_by_environment"];states={};sub={};firstself={}
with gzip.open(D/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"];states[e,t]=x
   for c in x["contacts"]:
    pair=(c["a_name"],c["b_name"])
    if c["impulses"][0]>1e-7 and any("shank" in q for q in pair) and any("rear_toe" in q for q in pair):
     if e not in firstself or t<firstself[e]["step"]:firstself[e]={"step":t,"pair":pair,"normal_impulse":c["impulses"][0]}
  elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4]);v=list(map(float,z[6:]));
   if v[4]>1e-7:sub.setdefault(e,[]).append((t,k,v[4]))
rows=[]
for e,end in enumerate(term):
 def sample(t):
  a=states[e,t];q=a["q"];x,y,z,w=q[3:7];pitch=math.asin(max(-1,min(1,2*(w*y-z*x))))
  return {"step":t,"position_m":q[:3],"pitch_rad":pitch,"velocity":a["v"][:6],"ground_pairs":sorted(set((c["a_name"],c["b_name"]) for c in a["contacts"] if c["impulses"][0]>1e-7 and "locomotion_ground" in (c["a_name"],c["b_name"])))}
 rows.append({"lane":e,"first_episode_steps":end,"termination_reason":ep["termination_reason_by_environment"][e],"first_recorded_shank_rear_toe_self_contact":firstself.get(e),"first_recorded_nonfoot_ground_substep":min(sub[e]) if e in sub else None,"initial":sample(1),"step_100":sample(100),"reported_final":sample(end),"following_endpoint":sample(end+1)})
out={"rows":rows,"scope":"Reason code 3 is forbidden external contact; nonfoot substep scalar is ground contact, while actual collider pair is sampled at endpoints. Episode-step counter and trace step differ at termination, so following-endpoint pairs cannot be asserted to be the first fatal substep. Shank/toe pairs are recorded self contacts and precede failure in these lanes, but causation is unproven. Records after first-episode termination are not success evidence."};(p/"failure-audit.json").write_text(json.dumps(out,indent=2)+"\n")
for r in rows: print(r["lane"],"end",r["first_episode_steps"],"self",r["first_recorded_shank_rear_toe_self_contact"],"first_ground",r["first_recorded_nonfoot_ground_substep"],"later_ground",r["following_endpoint"]["ground_pairs"],"pitch",r["reported_final"]["pitch_rad"])
