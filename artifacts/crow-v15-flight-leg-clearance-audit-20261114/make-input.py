from pathlib import Path
import gzip,hashlib,itertools,json
A=Path('artifacts/crow-v15-flight-leg-clearance-audit-20261114')
P=Path('artifacts/crow-v15-retract-hold-fix-20261114/push-window.json.gz')
x=json.load(gzip.open(P,'rt'))
s=[r for r in x['state'] if r['step']==385 and r['lane']==1]
assert len(s)==1
q=s[0]['q'];v=s[0]['v'];lims=[(-.85,.85),(-.1,1.55),(-.9,.7)]
records=[]
(A/'input.tsv').write_text('')
for delta in itertools.product((-.2,0.,.2),repeat=3):
 qq=q.copy()
 for i,d in enumerate(delta):
  qq[14+i]+=d;qq[17+i]+=d
 if not all(lo<=qq[k+i]<=hi for k in (14,17) for i,(lo,hi) in enumerate(lims)):continue
 n=len(records);records.append({'id':n,'delta':delta,'leg_angles':qq[14:20]})
 with (A/'input.tsv').open('a') as f:f.write('1 '+str(n)+' '+' '.join(map(str,qq+v))+'\n')
(A/'inputs.json').write_text(json.dumps({'source_window_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'source_raw_sha256':x['source_sha256'],'lane':1,'step':385,'root_q':q[:7],'records':records},indent=2)+'\n')
print('records',len(records),'original',next(r for r in records if r['delta']==(0.,0.,0.)))
