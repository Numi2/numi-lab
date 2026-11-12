from pathlib import Path
import gzip,hashlib,json,sys
A=Path("artifacts/crow-v15-contact-sensor-reset-control-20261112");no=Path(sys.argv[1]);reset=A/"candidate-20261126-rebound-separated/run.stderr.gz"
def h(p):
 z=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):z.update(b)
 return z.hexdigest()
def vals(p):
 d={}
 with gzip.open(p,"rt") as f:
  for line in f:
   if line.startswith("CROW_SENSOR_VALUE "):
    _,t,e,x=line.split();t=int(t);e=int(e)
    if e==7 and 248<=t<=260:d[str(t)]=float(x)
 return d
a=vals(no);b=vals(reset);assert all(a[str(t)]==b[str(t)] and a[str(t)]>0 for t in range(248,257));assert all(a[str(t)]>0 and b[str(t)]==0 for t in range(257,261))
(A/"sensor-no-reset-comparison.json").write_text(json.dumps({"source_commit":"d250f9b97d42883ae100d40f0e0c2efb5ca71821","source_path":"artifacts/crow-v15-dedicated-contact-sensor-20261112/candidate-20261126-rebound-phase/run.stderr.gz","source_sha256":h(no),"reset_trace_sha256":h(reset),"no_reset_lane7":a,"reset_lane7":b},indent=2)+"\n")
print("Lane 7 readings match through submission 256; the reset run then records invalid while the no-reset control remains active.")
