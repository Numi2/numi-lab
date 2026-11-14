from pathlib import Path
import gzip,json,math,hashlib
A=Path('artifacts/crow-v15-observer-timing-audit-20261114')
N=Path('artifacts/crow-v15-near-wing-transition-20261113')
R=Path('artifacts/crow-v15-launch-candidate-replay-20261114/push-20261126-rebound-lane7')
G=Path('artifacts/crow-v15-stronger-disturbance-20261114')
n=json.load(gzip.open(N/'push-window.json.gz','rt')); s={(x['lane'],x['step']):x for x in n['state']};c={(x[0],x[1],x[2]):x[3:] for x in n['contact']};a={(x[0],x[1],x[2]):x[3:] for x in n['aero']}
frames={x['step']:x for x in json.load(gzip.open(R/'replay.json.gz','rt'))['payload']['frames']};e=7
first=next(t for t in range(371,401) if all(sum(c[e,t,k][2:4])<=1e-7 for k in range(16)))
prior=max(abs(s[e,t]['q'][q]-s[e,370]['q'][q]) for t in range(371,first) for q in (9,10))
corrected=max(abs(s[e,t]['q'][q]-s[e,370]['q'][q]) for t in range(371,first+1) for q in (9,10))
replay=max(abs(frames[t]['q'][q]-frames[369]['q'][q]) for t in range(370,first) for q in (9,10))
imp=[a[e,t,k][3]*.02/16 for t in range(371,first) for k in range(16)]
ref={(x['lane'],x['step']):x for x in json.load(gzip.open(G/'straight-reference-states.json.gz','rt')) if x['lane'] in (6,7) and 900<=x['step']<=1101};g={}
with gzip.open(G/'gust-20261127-rebound-observer-gust/run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);
   if x['lane'] in (6,7) and 900<=x['step']<=1101:g[x['lane'],x['step']]=x

def pitch(q):
 x,y,z,w=q[3:7];return math.asin(max(-1,min(1,2*(w*y-z*x))))
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
rows=[]
for e in (6,7):
 wave=[]
 for frame in range(901,951):
  o=frame+1;x=g[e,o];y=ref[e,o];wave.append((frame,wrap(pitch(x['q'])-pitch(y['q'])),x['v'][4]-y['v'][4]))
 x=g[e,1101];y=ref[e,1101];pp=max(abs(v[1]) for v in wave);pv=max(abs(v[2]) for v in wave)
 rows.append({'lane':e,'accepted_frame_window':[901,950],'peak_abs_pitch_delta_rad':pp,'peak_abs_world_y_rate_delta_rad_s':pv,'response_threshold_met':pp>=.04 or pv>=.1,'pitch_delta_at_accepted_frame_1100_rad':wrap(pitch(x['q'])-pitch(y['q'])),'world_y_rate_delta_at_accepted_frame_1100_rad_s':x['v'][4]-y['v'][4]})
out={'scope':'timing reanalysis of development samples, not a qualified flight or takeoff','near_launch':{'lane':7,'first_fully_unloaded_contact_interval':first,'partially_loaded_interval_before':first-1,'preceding_interval_substeps_loaded':[k for k in range(16) if sum(c[7,first-1,k][2:4])>1e-7],'earlier_report_flap_displacement_rad':prior,'corrected_max_flap_displacement_through_end_of_last_partially_loaded_interval_rad':corrected,'independent_native_replay_displacement_rad':replay,'reference_observer_step':370,'reference_replay_frame':369,'enter_unloaded_observer_step':first,'enter_unloaded_replay_frame':first-1,'preceding_net_vertical_aero_impulse_Ns':sum(imp),'preceding_positive_vertical_aero_impulse_Ns':sum(v for v in imp if v>0),'preceding_negative_vertical_aero_impulse_Ns':sum(v for v in imp if v<0)},'strong_disturbance_corrected':rows,'caveats':['The replay alignment is directly established for the reproduced push lane 7; the same observer convention in other runs is inferred from shared host and needs independent validation.','Accepted final replay frame velocity is not independently observed.','Body transforms in the replay were not independently cross-checked against forward kinematics.']}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
