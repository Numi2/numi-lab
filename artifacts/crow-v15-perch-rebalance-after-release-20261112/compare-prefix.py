from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-perch-rebalance-after-release-20261112')
old=Path('artifacts/crow-v15-contact-friction-screen-20261111/high')
new=A/'high'
def rows(path):
 s={};finish={}
 with gzip.open(path/'trace.log.gz','rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=='step':
    e,t=map(int,z[1:3]);s[e,t]=z[z.index('q')+1:]
   elif z[0]=='event' and int(z[11]):finish.setdefault(int(z[1]),int(z[2]))
 return s,finish
r,rf=rows(old);c,cf=rows(new)
assert len(r)==len(c)==4400
def sub(path):
 d={}
 with gzip.open(path/'substeps.log.gz','rt') as f:
  for line in f:
   z=line.split()
   if z and z[0]=='CROW_TWO_TOE_SUBSTEP':
    t,k,e=map(int,z[1:4]);d[e,t-1,k]=z[4:]
 return d
x,y=sub(old),sub(new)
assert len(x)==len(y)==8*550*32
diff=[];counts=[0,0]
for e in range(8):
 end=549 if e<2 else rf.get(e,549)
 if rf.get(e)!=cf.get(e):diff.append([e,'different_return',rf.get(e),cf.get(e)])
 for t in range(end+1):
  counts[0]+=1
  if r[e,t]!=c[e,t]:diff.append([e,t,'qv'])
  for k in range(32):
   counts[1]+=1
   if x[e,t,k]!=y[e,t,k]:diff.append([e,t,k,'contact'])
res={'states_compared':counts[0],'substeps_compared':counts[1],'reference_return_steps':rf,'candidate_return_steps':cf,'differences':diff[:30],'difference_count':len(diff)}
(A/'parity.json').write_text(json.dumps(res,indent=2)+'\n')
print(res);assert not diff
