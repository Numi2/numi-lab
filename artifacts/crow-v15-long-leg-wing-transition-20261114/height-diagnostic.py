from pathlib import Path
import gzip,json,sys
D=Path(sys.argv[1]);rows={6:[],7:[]}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for l in f:
  if not l.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(l.split(' ',1)[1]);e=x['lane'];t=x['step']
  if e in rows and 450<=t<=1800:rows[e].append((t,x['q'][2],x['v'][2]))
out={}
for e,a in rows.items():
 bad=[x for x in a if x[1]<.25];minrow=min(a,key=lambda x:x[1]);out[e]={'first_height_below_floor':bad[0] if bad else None,'minimum_height':minrow,'count_below_floor':len(bad),'note':'state labels may lag same-run contact by one interval in these not-directly-checked lanes'}
(D/'height-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
