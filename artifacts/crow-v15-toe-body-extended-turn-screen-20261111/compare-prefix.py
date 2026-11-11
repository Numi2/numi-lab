from pathlib import Path
import gzip,json,math,subprocess,io,itertools
r=Path("artifacts");old=r/"crow-v15-toe-body-turn-screen-20261111";new=r/"crow-v15-toe-body-extended-turn-screen-20261111";manifest=json.loads((old/"stderr-parts.json").read_text())
import hashlib
for p in manifest["parts"]:
 b=(old/p["name"]).read_bytes();assert len(b)==p["bytes"] and hashlib.sha256(b).hexdigest()==p["sha256"]
proc=subprocess.Popen(["/bin/cat",*[str(old/x["name"]) for x in manifest["parts"]]],stdout=subprocess.PIPE)
def rows(f):
 for line in f:
  if line.startswith("CROW_FLIGHT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);
   if x["step"]<=1600:yield x
with gzip.GzipFile(fileobj=proc.stdout) as raw,io.TextIOWrapper(raw) as a,gzip.open(new/"flight.stderr.gz","rt") as b:
 n=0;mismatch=[]
 for x,y in itertools.zip_longest(rows(a),rows(b)):
  if x!=y:mismatch.append({"reference":(x["lane"],x["step"]) if x else None,"extended":(y["lane"],y["step"]) if y else None});break
  n+=1
proc.wait();assert proc.returncode==0
report={"exact_q_v_policy_samples":n,"first_mismatch":mismatch,"scope":"all logged samples through step 1600, not unsampled substeps"}
(new/"prefix-comparison.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
x=json.loads((new/"analysis.json").read_text())
for z in x["rows"]:
 if abs(abs(z["target_heading_rad"])-math.pi/2)>.01:continue
 vals=[s for s in z["samples"] if s["step"]>=450]
 good=[s["step"] for s in vals if abs(math.atan2(math.sin(z["target_heading_rad"]-s["yaw_rad"]),math.cos(z["target_heading_rad"]-s["yaw_rad"])))<.15]
 print(z["lane"],"good samples",good,"height",round(min(s["position"][2] for s in vals),3),round(max(s["position"][2] for s in vals),3),"max tilt",round(max(max(abs(s["pitch_rad"]),abs(s["roll_rad"])) for s in vals),3))
