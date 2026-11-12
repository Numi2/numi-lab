from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-ankle-pitch-response-screen-20261112");R=Path("artifacts/crow-v15-pitch-response-separated-20261111")
def read(p):
 x={}
 with gzip.open(p/"trace.log.gz","rt") as f:
  for l in f:
   z=l.split()
   if not z or z[0]!="step":continue
   e,t=map(int,z[1:3]);t+=1
   if t<=10:
    i=z.index("q");j=z.index("v");x[e,t]=[list(map(float,z[i+1:j])),list(map(float,z[j+1:]))]
 return x
ref=read(R/"zero");result={}
for arm in ("positive","negative"):
 vals=read(A/arm);bad=[];n=0
 for k in sorted(set(ref)&set(vals)):
  n+=1
  if ref[k]!=vals[k]:bad.append(k)
 result[arm]={"compared_samples":n,"mismatches":bad,"expected":80,"pass":n==80 and not bad}
(A/"prefix-comparison.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
