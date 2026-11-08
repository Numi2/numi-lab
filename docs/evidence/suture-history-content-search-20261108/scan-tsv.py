import hashlib,json,os,re,subprocess,sys
from pathlib import Path
w=Path(sys.argv[1]); root=Path(sys.argv[2]); env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0'); rows=[];errors=[];commit=None;pairs=set()
for line in (w/'tsv-history.txt').read_text().splitlines():
 if line.startswith('commit '):commit=line.split()[1]
 elif commit and line.lower().endswith('.tsv'):pairs.add((commit,line))
for commit,path in sorted(pairs):
 p=subprocess.run(['git','ls-tree','-z',commit,'--',path],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 split=p.stdout.rstrip(b'\0').split(b'\t',1)
 if p.returncode or len(split)!=2:
  errors.append({'commit':commit,'path':path,'code':p.returncode,'stderr':p.stderr.decode(errors='replace')});continue
 meta=split[0].decode().split()
 if len(meta)!=3 or split[1].decode(errors='surrogateescape')!=path:
  errors.append({'commit':commit,'path':path,'bad_tree':p.stdout.hex()});continue
 rows.append({'commit':commit,'path':path,'mode':meta[0],'kind':meta[1],'object':meta[2]})
ids=sorted({r['object'] for r in rows if r['mode']!='120000'})
p=subprocess.run(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],cwd=root,env=env,input=('\n'.join(ids)+'\n').encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError('batch-check: '+p.stderr.decode(errors='replace'))
checks={}
for line in p.stdout.decode().splitlines():
 fields=line.split(); checks[fields[0]]=fields[1:]
if set(checks)!=set(ids):raise RuntimeError('batch-check mismatch')
results=[]
for oid in ids:
 fields=checks[oid]
 if fields==['missing']:
  results.append({'object':oid,'status':'missing'});continue
 kind,size=fields[0],int(fields[1]); row={'object':oid,'kind':kind,'size':size}
 if kind!='blob' or size>8*1024*1024:
  row['status']='type_or_size_limit';results.append(row);continue
 p=subprocess.run(['git','cat-file','blob',oid],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode or len(p.stdout)!=size:
  row.update(status='read_error',code=p.returncode,stderr=p.stderr.decode(errors='replace'));results.append(row);continue
 row.update(status='read',sha256=hashlib.sha256(p.stdout).hexdigest(),schema=b'numi.dual-psm-suture-handoff-state.v3' in p.stdout,phase=b'tissue-receiver-bridge-start' in p.stdout)
 results.append(row)
res={'pairs':len(pairs),'errors':errors,'rows':rows,'objects':results}
(w/'tsv-results.json').write_text(json.dumps(res,indent=2)+'\n')
from collections import Counter
print('pairs',len(pairs),'resolved',len(rows),'errors',len(errors),'objects',len(results),'statuses',dict(Counter(r['status'] for r in results)))
matched={r['object'] for r in results if r.get('schema') or r.get('phase')}
print('matching_objects',len(matched))
for row in rows:
 if row['object'] in matched:print(row)
