from pathlib import Path
import gzip,json,math,sys
p=Path('artifacts/crow-v15-speed-command-screen-20261114')/sys.argv[1];ref={(x['lane'],x['step']):x for x in json.load(gzip.open('artifacts/crow-v15-flight-command-refactor-20261114/reference-states.json.gz','rt'))};cand={}
with gzip.open(p/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);cand[x['lane'],x['step']]=x
def forward(x):
 q=x['q'];v=x['v'];a,b,c,w=q[3:7];yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c));return math.cos(yaw)*v[0]+math.sin(yaw)*v[1]
rows=[]
for e in range(8):
 base=[forward(ref[e,t]) for t in range(1000,1201) if (e,t) in ref];new=[forward(cand[e,t]) for t in range(1000,1201) if (e,t) in cand];delta=(sum(new)/len(new)-sum(base)/len(base)) if new and base else None;rows.append({'lane':e,'command_m_s':.25 if e<4 else .45,'reference_count':len(base),'candidate_count':len(new),'reference_mean_yaw_frame_speed_m_s':sum(base)/len(base) if base else None,'candidate_mean_yaw_frame_speed_m_s':sum(new)/len(new) if new else None,'mean_delta_m_s':delta,'directional_authority_pass':len(base)==len(new)==201 and delta is not None and (delta<=-.01 if e<4 else delta>=.01)})
o={'rows':rows,'all_pass':all(x['directional_authority_pass'] for x in rows),'scope':'used-seed fixed-window directional effect, not setpoint accuracy or navigation'};(p/'speed-audit.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
