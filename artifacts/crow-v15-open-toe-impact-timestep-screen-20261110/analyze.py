from pathlib import Path
import gzip,json,importlib.util
spec=importlib.util.spec_from_file_location('prior','artifacts/crow-v15-separated-perch-feedback-20261110/analyze.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
D=Path('artifacts/crow-v15-open-toe-impact-timestep-screen-20261110');p={};g={};tr={};outcome={};contacts={}
with gzip.open(D/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='pose':p.setdefault((int(v[2]),int(v[3])),[]).append([float(z) for z in v[5:]])
  elif v[0]=='groundpose':g[int(v[2])]=[float(z) for z in v[3:]]
  elif v[0]=='trace':tr[(int(v[2]),int(v[3]))]=v[4:]
  elif v[0]=='case_result':outcome[int(v[2])]=' '.join(v[3:])
  elif v[0]=='contact':contacts[(int(v[2]),int(v[3]))]=contacts.get((int(v[2]),int(v[3])),0)+1
parity=all(tr.get((0,t))==prior.trace.get((1,0,t)) for t in range(10));rows=[]
for a in range(4):
 vals=[]
 for t in range(40):
  pp=p.get((a,t),[])
  if len(pp)==8 and a in g:vals.append((t,min(prior.gap(z,g[a]) for z in pp)))
 ang=[abs(float(q)) for t in range(40) if (a,t) in tr for q in tr[(a,t)][13:21]]
 rows.append({'arm':a,'outcome':outcome.get(a),'samples':len(vals),'first_overlap_step':next((t for t,x in vals if x<0),None),'first_over_half_mm_step':next((t for t,x in vals if x<-.0005),None),'minimum_gap_first_10_m':min((x for t,x in vals if t<10),default=None),'minimum_gap_m':min((x for t,x in vals),default=None),'max_abs_toe_rad':max(ang,default=None),'contact_endpoints':sum((a,t) in contacts for t in range(40))})
(D/'analysis.json').write_text(json.dumps({'original_first_ten_parity':parity,'rows':rows},indent=2)+'\n')
print('original first ten state parity',parity)
for r in rows:print(r)
