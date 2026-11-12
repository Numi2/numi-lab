from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-contact-sensor-reset-control-20261112");out={}
for arm in ("aliased","separated"):
 D=A/f"candidate-20261126-rebound-{arm}";summary=json.loads((D/"run.stdout").read_text());vals={}
 with gzip.open(D/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SENSOR_VALUE "):
    _,t,e,x=line.split();t=int(t);e=int(e)
    if e==7 and 248<=t<=260:vals[str(t)]=float(x)
 out[arm]={"host_requested_resets":summary["host_requested_resets"],"scheduled_resets":summary["scheduled_resets"],"lane7_sensor_values":vals}
assert out["separated"]["host_requested_resets"]>0 and out["separated"]["scheduled_resets"]
a=out["separated"]["lane7_sensor_values"];assert a["256"]>0 and all(a[str(t)]==0 for t in range(257,261))
(A/"sensor-reset-audit.json").write_text(json.dumps(out,indent=2)+"\n")
print("Nine host-requested resets; lane 7 recorded an active sensor at 256 and invalid value at 257-260.")
