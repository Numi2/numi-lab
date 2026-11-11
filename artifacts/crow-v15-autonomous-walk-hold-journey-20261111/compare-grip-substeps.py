from pathlib import Path
import gzip,json
D=Path("artifacts/crow-v15-autonomous-walk-hold-journey-20261111");paths={"journey":D/"autonomous-20261126/run.stderr.gz","reference":Path("artifacts/crow-v15-release-abort-latch-control-20261111/autonomous-20261126/run.stderr.gz")};prefixes=("CROW_TWO_TOE_SUBSTEP ","CROW_TOE_V_SUBSTEP ","CROW_SCENE_CONTACT ","CROW_PHASE_SAMPLE ");rows={}
for name,p in paths.items():
 records={k:[] for k in prefixes}
 with gzip.open(p,"rt") as f:
  for l in f:
   for k in prefixes:
    if l.startswith(k):
     x=l.split();step=int(x[1]);lane=int(x[2] if k=="CROW_PHASE_SAMPLE " else x[3])
     if lane==5 and 3500<=step<=3800:records[k].append(l.rstrip())
     break
 rows[name]=records
out={k.strip():{"journey_records":len(rows["journey"][k]),"reference_records":len(rows["reference"][k]),"exact_match":rows["journey"][k]==rows["reference"][k]} for k in prefixes}
(D/"grip-substep-comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
