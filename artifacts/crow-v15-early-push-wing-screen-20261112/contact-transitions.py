from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-early-push-wing-screen-20261112");out={}
for label in ("baseline","early","earlyinv"):
 rows=[];contacts={}
 with gzip.open((Path("artifacts/crow-v15-contact-timed-wing-screen-20261112/baseline-20261127-rebound") if label=="baseline" else A/(label+"-20261127-rebound"))/"run.stderr.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_TWO_TOE_SUBSTEP "):continue
   z=line.split();t,k,e=map(int,z[1:4])
   if 340<=t<=365:contacts[e,t,k]=sum(float(x) for x in z[8:10])>1e-7
 for e in range(8):
  seq=[(t,k,contacts[e,t,k]) for t in range(340,366) for k in range(16)];trans=[];last=seq[0][2]
  for t,k,v in seq[1:]:
   if v!=last:trans.append([t,k,"loaded" if v else "unloaded"]);last=v
  rows.append({"lane":e,"initially_loaded":seq[0][2],"transitions":trans})
 out[label]=rows
(A/"contact-transitions.json").write_text(json.dumps(out,indent=2)+"\n")
for name,rows in out.items():
 print(name,[(r["lane"],r["transitions"][:6]) for r in rows])
