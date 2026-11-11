from pathlib import Path
import json
A=Path('artifacts/crow-v15-passive-contact-resolution-20261111')
def read(n):
 rows=[[float(v) for v in l.split()] for l in (A/('steps'+str(n)+'.txt')).read_text().splitlines()];assert len(rows)==80 and all(int(v[0])==i and len(v)==20 for i,v in enumerate(rows));return rows
a,b=read(16),read(32);out={'samples':80,'first_10_max_root_x_difference_m':max(abs(x[1]-y[1]) for x,y in zip(a[:10],b[:10])),'first_10_max_root_z_difference_m':max(abs(x[2]-y[2]) for x,y in zip(a[:10],b[:10])),'first_10_max_rear_toe_difference_rad':max(abs(x[j]-y[j]) for x,y in zip(a[:10],b[:10]) for j in (5,6)),'first_10_group_summed_impulses':{str(n):[sum(r[8+3*g+j] for r in rows[:10]) for g in range(4) for j in (0,1)] for n,rows in ((16,a),(32,b))}};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
