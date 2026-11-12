from pathlib import Path
import gzip,json
src=Path("artifacts/crow-v15-rear-toe-damping-perch-screen-20261112");caps=json.loads((src/"cap-audit.json").read_text())["caps"];out={}
for arm in ("baseline","double","triple"):
 keys=sorted(set((z["lane"],z["interval"]) for z in caps[arm])); wanted={(e,t) for e,t0 in keys for t in range(t0-3,t0+2)};states={};controls={};events={};contacts={}
 with gzip.open(src/arm/"trace.log.gz","rt") as f:
  for line in f:
   z=line.split()
   if len(z)<3 or z[0] not in ("step","control","event","contact"):continue
   e,t=int(z[1]),int(z[2]);key=e,t
   if key not in wanted:continue
   if z[0]=="step":
    i=z.index("q");j=z.index("v");q=list(map(float,z[i+1:j]));v=list(map(float,z[j+1:]));states[key]={"right_rear_proximal_q":q[37],"right_rear_proximal_v":v[36],"right_rear_distal_q":q[38],"right_rear_distal_v":v[37]}
   elif z[0]=="control":controls[key]={"left_rear_target":float(z[3]),"right_rear_target":float(z[6]),"right_rear_previous_contact":float(z[8])}
   elif z[0]=="event":events[key]={"raw_event":z[3:]}
   elif "crow_right_rear_toe" in line and float(z[5])>0:
    contacts.setdefault(key,[]).append({"body_a":z[3],"body_b":z[4],"normal_impulse":float(z[5]),"separation":float(z[6])})
 rows=[]
 for e,t in keys:
  window=[]
  for step in range(t-3,t+2):
   key=e,step;assert key in states and key in controls and key in events
   window.append({"endpoint_step":step,**states[key],**controls[key],"right_rear_endpoint_contacts":contacts.get(key,[]),"raw_event":events[key]["raw_event"]})
  rows.append({"lane":e,"cap_interval_1based":t,"cap_substeps":[x["substep"] for x in caps[arm] if (x["lane"],x["interval"])==(e,t)],"window":window})
 out[arm]=rows
result={"arms":out,"scope":"Post-hoc windows around sampled -12 rad/s right rear proximal caps in constructed-arrival perch bench. Endpoint contact sets and control targets are not synchronized to each cap substep; logged commands are position targets, not measured torque. This does not identify a unique cause, qualify a landing, or measure corvid behavior."};Path("artifacts/crow-v15-perch-toe-cap-context-20261112/context.json").write_text(json.dumps(result,indent=2)+"\n")
for arm,rs in out.items():
 for x in rs:
  before=next(z for z in x["window"] if z["endpoint_step"]==x["cap_interval_1based"]-1)
  print(arm,x["lane"],x["cap_interval_1based"],"precap target",before["right_rear_target"],"q",before["right_rear_proximal_q"],"velocity",before["right_rear_proximal_v"],"rear contacts",len(before["right_rear_endpoint_contacts"]))
