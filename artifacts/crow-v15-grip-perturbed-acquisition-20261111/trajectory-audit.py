from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-grip-perturbed-acquisition-20261111');tr={}
with gzip.open(A/'results.txt.gz','rt') as f:
 for line in f:
  v=line.split()
  if v and v[0]=='trace':tr[int(v[1]),int(v[3])]=v[4:]
rows=[]
for lane in range(1,17):
 def sample(t):
  v=tr[lane,t];return {'step':t,'phase':int(v[0]),'root_x_m':float(v[6]),'root_height_m':float(v[7]),'root_vx_m_s':float(v[8]),'roll_quaternion_component':float(v[9]),'pitch_quaternion_component':float(v[10]),'root_vy_m_s':float(v[11]),'root_vz_m_s':float(v[12])}
 first=next((t for t in range(1500) if float(tr[lane,t][7])<0),None);rows.append({'lane':lane,'sample0':sample(0),'sample100':sample(100),'first_height_below_zero':sample(first) if first is not None else None,'sample1499':sample(1499)})
(A/'trajectory-audit.json').write_text(json.dumps({'lanes':rows,'scope':'constructed perched starts on an elevated isolated perch, not flight arrivals'},indent=2)+'\n');print('lanes below perch floor',sum(x['first_height_below_zero'] is not None for x in rows),'of',len(rows))
