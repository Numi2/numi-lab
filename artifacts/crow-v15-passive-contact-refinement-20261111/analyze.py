from pathlib import Path
import json
A=Path('artifacts/crow-v15-passive-contact-refinement-20261111');B=Path('artifacts/crow-v15-passive-contact-resolution-20261111')
def read(p,n):
 rows=[[float(v) for v in l.split()] for l in (p/('steps'+str(n)+'.txt')).read_text().splitlines()];assert len(rows)==80 and all(int(v[0])==i and len(v)==20 for i,v in enumerate(rows));return rows
data={n:read(A,n) for n in (16,32,64)};assert data[16]==read(B,16) and data[32]==read(B,32)
def compare(a,b):
 x,y=data[a],data[b];return {'max_root_x_m':max(abs(i[1]-j[1]) for i,j in zip(x[:10],y[:10])),'max_root_z_m':max(abs(i[2]-j[2]) for i,j in zip(x[:10],y[:10])),'max_rear_toe_angle_rad':max(abs(i[k]-j[k]) for i,j in zip(x[:10],y[:10]) for k in (5,6))}
r={'matched_reference_samples_per_resolution':80,'first_10':{'16_vs_32':compare(16,32),'32_vs_64':compare(32,64)},'first_10_summed_group_impulses':{str(n):[sum(row[8+3*g+j] for row in data[n][:10]) for g in range(4) for j in (0,1)] for n in data}};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
