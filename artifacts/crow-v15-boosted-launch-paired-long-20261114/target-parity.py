from pathlib import Path
import gzip,hashlib,json,subprocess,urllib.request,zlib
A=Path('artifacts/crow-v15-boosted-launch-paired-long-20261114'); V=Path('artifacts/crow-v15-boosted-launch-paired-verification-20261114'); ref=Path('artifacts/crow-v15-early-flight-boost-20261114'); d='push-20261126-rebound-longpaired'; m=json.loads((A/d/'stderr-parts.json').read_text()); v=json.loads((V/'push-verification.json').read_text()); tok=subprocess.check_output(['gh','auth','token'],text=True).strip(); z=zlib.decompressobj(31); h=hashlib.sha256(); n=0; buf=b''; targets={}
def parse(line, arm):
 if line.startswith(b'CROW_DRIVE_TARGET '):
  x=line.split(); t,e=map(int,x[1:3])
  if e in (6,7) and 335<=t<=500:
   key=(e,t); assert key not in targets[arm]; targets[arm][key]=list(map(float,x[3:]))
targets['push']={}
for part in m['parts']:
 url='https://api.github.com/repos/Numi2/numi-lab/contents/'+(A/d/part['file']).as_posix()+'?ref='+v['source_commit']; req=urllib.request.Request(url,headers={'Authorization':'Bearer '+tok,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'}); ph=hashlib.sha256(); size=0
 with urllib.request.urlopen(req,timeout=180) as f:
  while True:
   b=f.read(1<<20)
   if not b: break
   h.update(b); ph.update(b); n+=len(b); size+=len(b); buf+=z.decompress(b); lines=buf.split(b'\n'); buf=lines.pop()
   for line in lines: parse(line,'push')
 assert size==part['bytes'] and ph.hexdigest()==part['sha256']
assert n==m['bytes'] and h.hexdigest()==m['sha256'] and z.eof and not z.unused_data
if buf:parse(buf,'push')
targets['hold']={}
with gzip.open(A/'hold-20261126-rebound-longpaired/run.stderr.gz','rb') as f:
 for line in f: parse(line,'hold')
rows=[]
for e in (6,7):
 row={'lane':e}
 for arm in ('push','hold'):
  short=json.load(gzip.open(ref/(arm+'-window.json.gz'),'rt')); prior={(x[0],x[1]):x[2:] for x in short['target']}; missing=[t for t in range(335,501) if (e,t) not in prior or (e,t) not in targets[arm]]; diff=[t for t in range(335,501) if (e,t) in prior and (e,t) in targets[arm] and prior[e,t]!=targets[arm][e,t]]
  row[arm+'_missing_steps']=missing; row[arm+'_target_mismatch_steps']=diff
 for start,end in ((335,400),(335,500)):
  pairs=[abs(targets['push'][e,t][j]-targets['hold'][e,t][j]) for t in range(start,end+1) for j in (8,9,10,11) if (e,t) in targets['push'] and (e,t) in targets['hold']]
  row[f'wing_{start}_{end}']={'samples':len(pairs),'max_difference_rad':max(pairs) if pairs else None}
 rows.append(row)
out={'rows':rows,'push_source_commit':v['source_commit'],'push_download_sha256':h.hexdigest(),'scope':'Logged target equality only, not physical wing force equality; q/v and contact epochs separately audited.'}; (A/'target-parity.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
