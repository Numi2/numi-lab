from pathlib import Path
import gzip,hashlib,json,subprocess
A=Path("artifacts/crow-v15-downstroke-pushoff-screen-20261112"); C=A/"autonomous-20261126"; R=Path("artifacts/crow-v15-autonomous-walk-hold-journey-20261111/autonomous-20261126")
m=json.loads((R/"stderr-parts.json").read_text());h=hashlib.sha256()
for part in m["parts"]:
 b=(R/part["name"]).read_bytes();assert len(b)==part["bytes"] and hashlib.sha256(b).hexdigest()==part["sha256"];h.update(b)
assert h.hexdigest()==m["compressed_sha256"]
def read(p,parts=False):
 proc=subprocess.Popen(["cat",*[str(p/z["name"]) for z in m["parts"]]],stdout=subprocess.PIPE) if parts else None
 f=gzip.open(proc.stdout if parts else p/"run.stderr.gz","rt");sup={};sub={};caps=[];errors=[]
 try:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t,e=x["step"],x["lane"]
    if t<=500:
     assert (e,t) not in sup;sup[e,t]=x
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if t<=500:
     assert int(z[4])<=int(z[5]);assert (e,t,k) not in sub;sub[e,t,k]=list(map(float,z[6:]))
   elif l.startswith("CROW_TOE_V_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if t<=500 and max(abs(float(x)) for x in z[4:])>=11.999:caps.append((e,t,k))
   elif "CROW_SUBSTEP_ERROR" in l: errors.append(l.strip())
 finally:
  f.close()
  if proc:proc.wait()
 return sup,sub,caps,errors
ref,rs,rc,re=read(R,True);cur,cs,cc,ce=read(C)
assert len(cur)==len(ref)==4000 and len(cs)==len(rs)==64000
ends=json.loads((C/"run.stdout").read_text())["first_episode_navigation"]["steps_by_environment"]
rows=[]
for e in range(8):
 dif=[t for t in range(1,501) if any(ref[e,t][k]!=cur[e,t][k] for k in ("q","v","policy_action","contacts"))]
 cdif=[(t,k) for t in range(1,501) for k in range(16) if rs[e,t,k]!=cs[e,t,k]]
 def interval(sub,t):return [sum(sub[e,t,k][j] for k in range(16)) for j in range(5)]
 def unload(sub,sup):
  return next((t for t in range(340,400) if all(sum(interval(sub,u)[2:4])<=1e-7 and sup[e,u]["v"][2]>0 for u in (t,t+1))),None)
 rt=unload(rs,ref);ct=unload(cs,cur)
 def impulse(sub,t):return [sum(interval(sub,u)[i] for u in range(340,t+1)) for i in (0,1)] if t else None
 rows.append({"lane":e,"first_state_difference":dif[:1],"prefix_states_equal":not any(t<=344 for t in dif),"first_contact_substep_difference":cdif[:1],"prefix_contacts_equal":not any(t<=344 for t,k in cdif),"reference_unloaded_rising_interval":rt,"candidate_unloaded_rising_interval":ct,"reference_foot_upward_impulse":impulse(rs,rt),"candidate_foot_upward_impulse":impulse(cs,ct),"candidate_max_nonfoot_interval_impulse":max(interval(cs,t)[4] for t in range(340,361)),"candidate_toe_caps":[[t,k] for lane,t,k in cc if lane==e and 340<=t<=500],"candidate_height_step500":cur[e,500]["q"][2],"candidate_first_episode_steps":ends[e]})
out={"rows":rows,"reference_errors":re,"candidate_errors":ce,"candidate_exit":json.loads((C/"run-result.json").read_text())["exit_code"],"limitations":"Foot impulse is summed normal vertical component over steps 340 to first unloaded interval, not isolated muscle force. No measured contact-point slip or substep wing orientation. Development seed only."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out,indent=2))
