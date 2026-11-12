from pathlib import Path
import gzip,json
r=Path("artifacts");old=r/"crow-v15-negative-ankle-matched-control-20261112";new=r/"crow-v15-dedicated-contact-sensor-20261112";out={}
def read(path):
 d={}
 with gzip.open(path,"rt") as f:
  for line in f:
   if line.startswith("CROW_SENSOR_VALUE "):
    _,t,e,x=line.split();t=int(t);e=int(e)
    if 202<=t<=300 and 2<=e<8:
     assert (e,t) not in d;d[e,t]=float(x)
 return d
for seed in (20261126,20261127):
 a=read(old/f"candidate-{seed}-rebound-phase/run.stderr.gz");b=read(new/f"candidate-{seed}-rebound-phase/run.stderr.gz");diff=[k for k in a.keys() & b.keys() if a[k]!=b[k]]
 out[str(seed)]={"records_old":len(a),"records_new":len(b),"missing":len(a.keys()-b.keys()),"extra":len(b.keys()-a.keys()),"different":len(diff),"first_difference":diff[0] if diff else None}
(new/"sensor-parity.json").write_text(json.dumps(out,indent=2)+"\n")
for seed,x in out.items():
 assert x["records_old"]==x["records_new"]==594 and x["missing"]==x["extra"]==x["different"]==0,(seed,x)
print("All 594 active-window sensor values match on each seed.")
