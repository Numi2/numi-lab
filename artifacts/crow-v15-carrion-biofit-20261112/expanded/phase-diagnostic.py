from pathlib import Path
import json, numpy as np, hashlib
A=Path(__file__).resolve().parent;v=json.loads((A/'sim-flap-velocity.json').read_text());states=json.loads((A/'sim-tail-states.json').read_text());fit=json.loads((A/'fit-training.json').read_text()); name=fit['selected'];flap=v[name]['rows'];tail=states[name]['states'];assert len(flap)==len(tail)==1600
freq=np.fft.rfftfreq(250,.02);band=(freq>=1)&(freq<=12)
def ps(x):
 x=np.asarray(x);x=x-x.mean(axis=0);y=np.fft.rfft(x*np.hanning(250)[:,None],axis=0);p=np.sum(abs(y)**2,axis=1);i=np.flatnonzero(band)[np.argmax(p[band])];return float(freq[i])
rows=[]
for start in (450,700,950,1200):
 i=start-1; f=np.array([[x['left_flap_rad_s'],x['right_flap_rad_s']] for x in flap[i:i+250]]);vworld=np.array([x['velocity_world_m_s'] for x in tail[i-1:i+251]]);az=(vworld[2:,2]-vworld[:-2,2])/.04;energy=(f*f).mean(axis=1);r=[]
 for lag in range(-10,11):
  x=energy[max(0,lag):min(250,250+lag)];y=az[max(0,-lag):min(250,250-lag)];r.append((lag,float(np.corrcoef(x,y)[0,1])))
 best=max(r,key=lambda x:abs(x[1]));rows.append({'start_step':start,'flap_velocity_peak_hz':ps(f),'squared_flap_velocity_peak_hz':ps(energy[:,None]),'tail_world_vertical_acceleration_peak_hz':ps(az[:,None]),'largest_absolute_correlation_over_plus_minus_0_2s':{'lag_steps':best[0],'correlation':best[1]}})
out={'selected':name,'rows':rows,'inputs':{n:hashlib.sha256((A/n).read_bytes()).hexdigest() for n in ('sim-flap-velocity.json','sim-tail-states.json','fit-training.json')},'scope':'Post-hoc spectral and lagged association, not force attribution: phase-harmonic structure can occur for several reasons; squared joint velocity is only one factor in the unsteady wing closure. Four dependent windows from one simulated lane; not crow kinematics.'}
(A/'phase-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
