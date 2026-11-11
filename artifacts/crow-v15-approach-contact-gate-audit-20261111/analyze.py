from pathlib import Path
import gzip,collections,json,math
D=Path("artifacts/crow-v15-approach-contact-gate-audit-20261111");R=Path("artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126");ends=json.loads((R/"run.stdout").read_text())["first_episode_navigation"]["steps_by_environment"];phase={};state={};counts={e:collections.Counter() for e in range(8)}
with gzip.open(R/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_PHASE_SAMPLE "):
   z=l.split();t,e=int(z[1]),int(z[2]);
   if 3000<=t<=ends[e]:phase[e,t]=[float(x) for x in z[4:]]
  elif l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1]);t,e=x["step"],x["lane"]
   if 3000<=t<=ends[e]:state[e,t]=x
for (e,t),z in phase.items():
 if not (.5<z[0]<1.5) or (e,t) not in state:continue
 v=state[e,t]["v"];tests=[z[1]>.05 and z[2]>.05,abs(z[3]-.012)<.025,abs(z[4])<.16,math.hypot(v[0],v[1])<.08,abs(v[2])<.12];c=counts[e];c["phase_one_samples"]+=1
 for i,flag in enumerate(tests):c["pass_proxy_"+str(i)]+=flag
 for i in range(5):c["pass_all_but_proxy_"+str(i)]+=all(tests[:i]+tests[i+1:])
 c["pass_all_proxies"]+=all(tests)
rows=[{"lane":e,**dict(counts[e])} for e in range(8)];out={"rows":rows,"tests":["both logged foot loads > 0.05", "absolute final-goal x error minus 0.012 < 0.025 m", "absolute final-goal y error < 0.16 m", "horizontal root speed < 0.08 m/s", "absolute vertical speed < 0.12 m/s"],"limitations":"Logged phase, task-state loads and root endpoint velocity are read at the same numbered observer step. They are proxies for the actual predicate, which uses current compact contacts and prior task state in different kernels; timing and contact-group mapping are not proven identical. Phase transition itself can occur before a same-numbered sample. This only identifies where to instrument and is not causal proof."};(D/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
