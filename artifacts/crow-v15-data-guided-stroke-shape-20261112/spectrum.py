from pathlib import Path
import gzip,json,hashlib,numpy as np
A=Path("artifacts/crow-v15-data-guided-stroke-shape-20261112"); training=json.loads(Path("artifacts/crow-v15-carrion-biofit-20261112/expanded/fit-training.json").read_text())["training_subjects"]
def rotate_inv(q,v):
 a=-np.array(q[:3]);w=q[3];return v+2*np.cross(a,np.cross(a,v)+w*v)
def measure(a):
 p=np.fft.rfft((a-a.mean(axis=0))*np.hanning(250)[:,None],axis=0);power=np.sum(np.abs(p)**2,axis=1);freq=np.fft.rfftfreq(250,.02);mask=(freq>=1)&(freq<=12);i=np.flatnonzero(mask)[np.argmax(power[mask])];peak=float(freq[i]);return {"peak_hz":peak,"concentration":float(power[(abs(freq-peak)<=.40000001)&mask].sum()/power[mask].sum())}
results=[]
for name in ("sham","warped"):
 path=A/(name+"-replay.json.gz");doc=json.load(gzip.open(path,"rt"));p=doc["payload"];frames=p["frames"];assert p["timestep_seconds"]==.02 and p["body_state_stride"]==13 and not p["scheduled_resets"];index=p["body_names"].index("crow_tail");assert len(frames)==1600 and [f["step"] for f in frames]==list(range(1,1601));states=[f["body_states"][index*13:(index+1)*13] for f in frames];windows=[]
 for start in (450,700,950,1200):
  data=[]
  for step in range(start,start+250):
   i=step-1;assert not frames[i]["done"];acc=(np.array(states[i+1][7:10])-np.array(states[i-1][7:10]))/.04+np.array([0,0,9.81]);data.append(rotate_inv(states[i][3:7],acc)/9.81)
  windows.append({"start_step":start,**measure(np.array(data))})
 peak=float(np.median([w["peak_hz"] for w in windows]));conc=float(np.median([w["concentration"] for w in windows]));score=float(np.median([abs(peak-b["peak_hz"])+abs(conc-b["concentration"])/.25 for b in training]));results.append({"name":name,"windows":windows,"median_peak_hz":peak,"median_concentration":conc,"training_score":score,"replay_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
out={"results":results,"improved_training_objective":results[1]["training_score"]<results[0]["training_score"],"scope":"One development seed and lane; rigid tail COM proxy versus carrion-crow tail-feather sensor. Behavioral label includes flying other than flapping. Similarity is not biological or flight qualification."};(A/"spectrum.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
