from pathlib import Path
import gzip,json,subprocess
A=Path("artifacts/crow-v15-early-wing-takeoff-screen-20261112");C=A/"autonomous-20261126";R=Path("artifacts/crow-v15-autonomous-walk-hold-journey-20261111/autonomous-20261126")
m=json.loads((R/"stderr-parts.json").read_text());groups={}
for label,p in (("candidate",C),("baseline",R)):
 proc=subprocess.Popen(["cat",*[str(p/x["name"]) for x in m["parts"]]],stdout=subprocess.PIPE) if label=="baseline" else None
 f=gzip.open(proc.stdout if proc else p/"run.stderr.gz","rt");sub={}
 for l in f:
  if l.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4])
   if 348<=t<=354:sub[e,t,k]=list(map(float,z[6:]))
 f.close()
 if proc:proc.wait()
 assert len(sub)==8*7*16;groups[label]=sub
rows=[]
for e in range(8):
 row={"lane":e}
 for label,sub in groups.items():
  loaded=[(t,k) for t in range(348,355) for k in range(16) if sum(sub[e,t,k][2:4])>1e-7]
  assert loaded;last=loaded[-1]
  samples=[sub[e,t,k] for t in range(348,last[0]+1) for k in range(16) if t<last[0] or k<=last[1]]
  impulses=[sum(z[i] for z in samples) for i in (0,1)];weight=.45006*9.81*.00125*len(samples)
  row[label]={"last_loaded_substep":last,"duration_ms":len(samples)*1.25,"foot_upward_impulse_Ns":impulses,"weight_impulse_Ns":weight,"ground_impulse_minus_weight_Ns":sum(impulses)-weight}
 rows.append(row)
out={"rows":rows,"limits":"Window starts at interval 348; 0.45006 kg is model mass. Sum of ground vertical impulse minus weight is a force-balance diagnostic, not isolated leg muscle impulse or biological target. Wing and passive aerodynamic forces remain."};(A/"push-analysis.json").write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out,indent=2))
