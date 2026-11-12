from pathlib import Path
import hashlib,json,numpy as np
A=Path(__file__).resolve().parent
index=json.loads((A/'source-sample-index.json').read_text()); src=json.loads((A/'sim-tail-states.json').read_text())
ordered=['crow-v15-sweep-pitch-feedback-screen-20261112','crow-v15-bank-pitch-feedback-screen-20261112','crow-v15-flight-sideslip-feedback-screen-20261112','crow-v15-forward-pitch-feedback-screen-20261112']
subjects=[]
for i in range(8):
 rows=[r for r in index['rows'] if r['individual_id']==i]
 if rows:subjects.append({'individual_id':i,'count':len(rows),'peak_hz':float(np.median([r['peak_hz'] for r in rows])),'concentration':float(np.median([r['band_concentration'] for r in rows]))})
def rotate_inv(q,v):
 a=-np.array(q[:3]);w=q[3];return v+2*np.cross(a,np.cross(a,v)+w*v)
def measure(a):
 p=np.fft.rfft((a-a.mean(axis=0))*np.hanning(250)[:,None],axis=0); power=np.sum(np.abs(p)**2,axis=1); freq=np.fft.rfftfreq(250,.02); mask=(freq>=1)&(freq<=12);i=np.flatnonzero(mask)[np.argmax(power[mask])];peak=float(freq[i]);return {'peak_hz':peak,'concentration':float(power[(abs(freq-peak)<=.40000001)&mask].sum()/power[mask].sum())}
results=[]
for name in ordered:
 item=src[name];s=item['states'];assert [x['step'] for x in s]==list(range(1,1601)); windows=[]
 for start in (450,700,950,1200):
  data=[]
  for step in range(start,start+250):
   i=step-1;assert not s[i]['done']
   acc=(np.array(s[i+1]['velocity_world_m_s'])-np.array(s[i-1]['velocity_world_m_s']))/.04+np.array([0,0,9.81]);data.append(rotate_inv(s[i]['orientation_xyzw'],acc)/9.81)
  windows.append({'start_step':start,**measure(np.array(data))})
 peak=float(np.median([v['peak_hz'] for v in windows]));conc=float(np.median([v['concentration'] for v in windows]));score=float(np.median([abs(peak-b['peak_hz'])+abs(conc-b['concentration'])/.25 for b in subjects]));results.append({'name':name,'windows':windows,'median_peak_hz':peak,'median_concentration':conc,'training_score':score})
assert len(subjects)==8
winner=min(results,key=lambda x:x['training_score']);out={'training_subjects':subjects,'candidates':results,'selected':winner['name'],'objective':'median per-individual abs(peak difference)/1 Hz + abs(concentration difference)/0.25','scope':'Exploratory model ranking only, uncalibrated tail-COM sensor proxy; different species, broad behavior label, one simulated start per candidate, and no controller or flight qualification. No held-out subjects were used for selection.','input_sha256':{n:hashlib.sha256((A/n).read_bytes()).hexdigest() for n in ('PLAN.md','source-sample-index.json','sim-tail-states.json')}}
(A/'fit-training.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
