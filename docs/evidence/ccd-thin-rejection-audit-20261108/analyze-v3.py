#!/usr/bin/env python3
import argparse,hashlib,json,math,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("archive",type=Path);a=p.parse_args()
with tarfile.open(a.archive,"r:gz") as t:
 observations=json.load(t.extractfile("ccd-useful-motion005/world-steps.json"))["runs"]
 lines=t.extractfile("ccd-useful-motion005/world-steps.stderr").read().decode().splitlines()
cases={}; key=None
for line in lines:
 if line.startswith("ccd_case="):
  q=json.loads(line.split("=",1)[1]);key=(q["limit"],q["mode"],q["fixture"],q["scale"]);cases[key]=[]
 elif line.startswith("ccd_event_trace=") and key is not None:cases[key].append(json.loads(line.split("=",1)[1]))
def boundary(e,extents):
 q=e["q"];v=e["v"];qn=math.sqrt(sum(z*z for z in q[3:7]));x,y,z,w=(z/qn for z in q[3:7]);hx,hy,hz=extents
 nyx=2*(x*y+z*w);nyy=1-2*(x*x+z*z);nyz=2*(y*z-x*w)
 gap=q[1]-hx*abs(nyx)-hy*abs(nyy)-hz*abs(nyz)
 points=[]
 for c in e["contacts"]:
  r=[c["point"][i]-q[i] for i in range(3)]
  vn=v[1]+v[5]*r[0]-v[3]*r[2]
  points.append({"recorded_gap_m":c["gap"],"recorded_pre_normal_velocity_m_s":c["pre_vn"],"reported_normal_impulse_N_s":c["J"],"reconstructed_post_normal_velocity_m_s":vn})
 return {"pass":e["pass"],"status":e["status"],"unresolved":e["unresolved"],"pair_flags":e["pair_flags"],"pair_work":e["pair_work"],"elapsed_s":e["time"][0],"remaining_s":e["time"][1],"support_gap_m":gap,"contacts":points}
rows=[]
for r in observations:
 if r["mode"]!=0 or r["limit"] not in (16,128) or r["fixture"] not in (1,2,12,14):continue
 k=(r["limit"],0,r["fixture"],r["scale"]);events=cases[k]
 assert len(events)==8 and [e["pass"] for e in events]==[0,1,2,3,0,1,2,3]
 assert events[:4]==events[4:] and r["repeat_identical"] and r["rejected_state_restored"]
 first=events[:4]; index=next((i for i,e in enumerate(first) if e["status"]!=0),None)
 thin=r["fixture"] in (12,14)
 assert (index is not None)==thin and r["accepted"]!=thin
 assert not thin or r["rejected_contact_cache_empty"]
 prev=boundary(first[index-1],r["half_extents"]) if index is not None else None
 if thin and prev: assert prev["status"]==0 and prev["unresolved"]==0 and prev["contacts"] and prev["remaining_s"]>0
 rows.append({"limit":r["limit"],"fixture":r["fixture"],"scale":r["scale"],"accepted":r["accepted"],"byte_replay_and_rejection_restored":True,"prior_boundary":prev,"failure_boundary":boundary(first[index],r["half_extents"]) if index is not None else None,"failure_kind":("unresolved_cast" if index is not None and first[index]["unresolved"] else "event_budget_exhausted" if index is not None and first[index]["status"]==9 else "other_failure" if index is not None else None)})
assert len(rows)==16
print(json.dumps({"schema":"numi.ccd.thin-rejection-audit.v1","archive_sha256":hashlib.sha256(a.archive.read_bytes()).hexdigest(),"cases":rows,"limits":["Reconstructed point velocities use recorded contact midpoint, candidate center and world angular velocity, and assume the plane normal is world +Y.","The candidate trace describes rejected trial paths; a negative post-response velocity does not by itself establish failure of the discrete contact target.","The trace contains no explicit certificate branch reason, so unresolved casts may not be classified more narrowly solely from pair flags."]},indent=2))
