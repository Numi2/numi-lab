from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-first-contact-load-audit-20261111');B=Path('artifacts/crow-v15-front-toe-perturbation-20261111')
def read(p):
 traces={};groups={}
 with gzip.open(p,'rt') as f:
  for line in f:
   v=line.split()
   if v and v[0]=='trace':traces[(int(v[1]),int(v[3]))]=v[4:]
   if v and v[0]=='contact_group':groups[(int(v[1]),int(v[3]),int(v[4]))]=list(map(float,v[5:8]))
 return traces,groups
ref,_=read(B/'results.txt.gz');reports=[]
for case,lane in [(1,1),(0,15)]:
 tr,g=read(A/('case-'+str(case)+'.txt.gz'));diff=[t for t in range(150) if tr.get((case,t))!=ref.get((lane,t))];assert len(tr)==150 and not diff,(case,diff[:10]);windows=[]
 for lo,hi in [(0,10),(10,30),(30,60),(60,100)]:
  groups=[]
  for k in range(4):
   vals=[g[(case,t,k)] for t in range(lo,hi)];groups.append([sum(row[j] for row in vals)/(hi-lo) for j in range(3)])
  windows.append({'ticks':[lo,hi-1],'mean_group_impulses_xyz':groups})
 reports.append({'case':case,'reference_lane':lane,'trace_differences':len(diff),'windows':windows})
(A/'analysis.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
