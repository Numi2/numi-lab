#!/usr/bin/env python3
"""FP64 check of every captured literal free-body segment, without integration."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("observations",type=Path);p.add_argument("trace",type=Path);a=p.parse_args()
runs=json.loads(a.observations.read_text())["runs"];cases={};key=None
for line in a.trace.open():
 if line.startswith("ccd_case="):
  x=json.loads(line.split("=",1)[1]);key=(x["limit"],x["mode"],x["fixture"],x.get("scale",1));cases.setdefault(key,[])
 elif line.startswith("ccd_event_trace=") and key is not None:
  cases[key].append(json.loads(line.split("=",1)[1]))
def segment_gap(a,b,extents):
 a=np.asarray(a,float);b=np.asarray(b,float);q0=a[3:7];q1=b[3:7]
 q0=q0/np.linalg.norm(q0);q1=q1/np.linalg.norm(q1)
 if q0@q1<0:q1=-q1
 theta=np.arccos(np.clip(q0@q1,-1,1));t=np.linspace(0,1,100001)
 q=(np.sin((1-t)*theta)[:,None]*q0+np.sin(t*theta)[:,None]*q1)/np.sin(theta) if theta>1e-9 else np.tile(q0,(len(t),1))
 # Y row of R(q); the exact support of an oriented box against Y=0.
 nx=2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]);ny=1-2*(q[:,0]**2+q[:,2]**2);nz=2*(q[:,1]*q[:,2]-q[:,0]*q[:,3])
 gap=a[1]+t*(b[1]-a[1])-extents[0]*np.abs(nx)-extents[1]*np.abs(ny)-extents[2]*np.abs(nz)
 # A Lipschitz bound encloses the entire interval between FP64 samples.
 speed=np.linalg.norm(b[:3]-a[:3])+np.linalg.norm(extents)*(2*theta)
 return float(gap.min()),float(gap.min()-speed/(2*(len(t)-1))-2e-7)
rows=[]
for run in runs:
 if not run["accepted"]:continue
 key=(run["limit"],run["mode"],run["fixture"],run.get("scale",1));trace=cases[key];events=trace[:4]
 initial=[0,run["initial_q_y"],0,0,0,0,1]
 values=[];previous=initial;elapsed=0
 for e in events:
  if e["time"][0]>elapsed+1e-10:
   sampled,bound=segment_gap(previous,e["q"],run["half_extents"])
   dt=e["time"][0]-elapsed;acceleration=np.asarray(e["a"],float)
   path_error=dt*dt*.25*(np.linalg.norm(acceleration[:3])+np.linalg.norm(run["half_extents"])*np.linalg.norm(acceleration[3:]))
   values.append({"pass":e["pass"],"sample_min_gap_m":sampled,"discrete_acceleration_path_envelope_m":float(path_error),"whole_interval_lower_bound_m":bound-path_error})
   elapsed=e["time"][0];previous=e["q"]
 rows.append({"limit":run["limit"],"mode":run["mode"],"fixture":run["fixture"],"scale":run.get("scale",1),"completed_seconds":elapsed,"segments":values,"qualified":bool(values) and elapsed>=.01-1e-8 and all(v["whole_interval_lower_bound_m"]>=-1e-6 for v in values)})
print(json.dumps({"schema":"numi.ccd.segment-geometry.v1","accepted_runs":rows,"all_accepted_segments_qualified":all(r["qualified"] for r in rows),
 "method":"100001 FP64 endpoint-geodesic/linear samples per literal segment, whole-interval Lipschitz enclosure, H^2/4 discrete acceleration envelope and 0.2 micrometre rounding guard.",
 "limits":["This observer is limited to the force-free single floating box fixtures; it does not certify arbitrary articulated link interpolation.",
 "Rejected-step trial segments are not accepted trajectories."]},indent=2))
