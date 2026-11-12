from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-wing-roll-feedback-screen-20261112')
P=Path('artifacts/crow-v15-reversed-wing-bank-screen-20261112')
def read(path):
 d={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);d[x['lane'],x['step']]=x
 return d
old=read(P/'bank.stderr.gz');new=read(A/'bank.stderr.gz');prefix=[k for k in new if k[1]<450]
assert len(prefix)==48 and all(k in old and all(old[k][z]==new[k][z] for z in ('q','v','policy_action')) for k in prefix)
analysis=json.loads((A/'bank-analysis.json').read_text());rows=[]
for row in analysis['rows']:
 s=[x for x in row['samples'] if x['step']>=450];last=s[-1];vx,vy,_=last['velocity'];track=math.atan2(vy,vx);diff=math.atan2(math.sin(track-last['yaw_rad']),math.cos(track-last['yaw_rad']))
 rows.append({'lane':row['lane'],'target_max_bank_rad':[.15,.30,.45][row['lane']//2],'terminal_step':row['first_episode_steps'],'heading_error_rad':row['final_sample_heading_error_rad'],'max_abs_sampled_roll_rad':max(abs(x['roll_rad']) for x in s),'sample_height_range_m':[min(x['position'][2] for x in s),max(x['position'][2] for x in s)],'final_ground_speed_m_s':math.hypot(vx,vy),'final_track_minus_heading_rad':diff,'toe_caps':row['toe_caps']})
out={'matched_pre_turn_samples':len(prefix),'pre_turn_mismatches':0,'rows':rows,'scope':'Sparse logged snapshots only; no continuous extrema or biological bank calibration. Direct body moments persist through the assisted launch, then are off from 9 seconds. Used development seed; not qualified flight.'}
(A/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
