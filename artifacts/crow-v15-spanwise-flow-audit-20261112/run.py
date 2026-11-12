from pathlib import Path
import hashlib,json,os,subprocess,statistics,math
R=Path.cwd();A=R/'artifacts/crow-v15-spanwise-flow-audit-20261112';B=R/'build/crow-v15-spanwise-flow-audit-20261112';P=R/'artifacts/crow-v15-aero-yaw-sensitivity-20261112';L=R/'build/crow-v15-toe-velocity-substep-observer-20261111';exe=R/'build/crow-v15-aero-yaw-sensitivity-20261112/evaluate'
lines=[x for x in (P/'input.txt').read_text().splitlines() if x.split()[0]=='0'];assert len(lines)==72
(A/'input.txt').write_text('\n'.join(lines)+'\n'); env={**os.environ,'DYLD_LIBRARY_PATH':str(L)}
for name,ml in (('current',R/'build/crow-v15-wing-steering-without-body-moments-20261112/baseline.metallib'),('sectional',B/'sectional.metallib')):
 for suffix in ('','-repeat'):
  with (A/'input.txt').open('rb') as f,(A/(name+suffix+'.txt')).open('wb') as out,(A/(name+suffix+'.stderr')).open('wb') as err: subprocess.run([str(exe),str(ml)],stdin=f,stdout=out,stderr=err,env=env,check=True)
 assert (A/(name+'.txt')).read_bytes()==(A/(name+'-repeat.txt')).read_bytes()
old=[x for x in (P/'output.txt').read_text().splitlines() if x.split()[0]=='0'];cur=(A/'current.txt').read_text().splitlines();assert cur==old, 'current kernel does not reproduce prior results'
other=(A/'sectional.txt').read_text().splitlines();assert len(other)==len(cur)==72
rows=[]
for x,y in zip(cur,other):
 a=x.split();b=y.split();assert a[:3]==b[:3] and len(a)==len(b)==12
 u=list(map(float,a[3:]));v=list(map(float,b[3:]));assert all(map(math.isfinite,u+v))
 def sums(z):return [z[0]+z[4],z[1]+z[5],z[2]+z[6],z[3]+z[7]]
 c=sums(u);s=sums(v);rows.append({'lane':int(a[1]),'step':int(a[2]),'current_wings':[round(n,10) for n in c],'sectional_wings':[round(n,10) for n in s],'difference':[round(q-r,10) for q,r in zip(s,c)]})
def stats(i):
 a=[abs(row['difference'][i]) for row in rows];a.sort();return {'median_abs':statistics.median(a),'p95_abs':a[int(.95*(len(a)-1))],'max_abs':max(a)}
out={'sample_count':len(rows),'original_matches_prior':True,'both_repeat_identically':True,'absolute_difference_statistics':dict(zip(('force_x_N','force_y_N','force_z_N','root_yaw_moment_Nm'),map(stats,range(4)))),'rows':rows,'scope':'Dependent sparse snapshots; counterfactual modeling comparison only. Not net body load, integrated flight, aerodynamic calibration or controller qualification.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
