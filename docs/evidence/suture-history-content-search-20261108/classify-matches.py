import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
w=Path(sys.argv[1]);root=Path(sys.argv[2]);env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0');rows=[];errors=[]
for rec in map(json.loads,(w/'archive-content.jsonl').read_text().splitlines()):
 if not rec['matches']:continue
 targets={m['name']:m for m in rec['matches']}
 if any(m['size']>1048576 for m in targets.values()):
  errors.append({'object':rec['object'],'error':'member larger than 1 MiB'});continue
 p=subprocess.Popen(['git','-C',str(root),'cat-file','blob',rec['object']],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 found=set()
 try:
  with tarfile.open(fileobj=p.stdout,mode='r|*') as t:
   for m in t:
    if m.name not in targets:continue
    f=t.extractfile(m)
    if not f:raise ValueError('cannot read '+m.name)
    b=f.read(1048577)
    if len(b)!=m.size:raise ValueError('size mismatch '+m.name)
    fields={}
    for line in b.decode('utf8').splitlines():
     k,sep,v=line.partition('\t')
     if sep and (k in ('schema','phase','step') or (k.startswith('matter_') and ('path' in k or 'hash' in k))):fields[k]=v
    rows.append({'object':rec['object'],'archive_paths':rec['paths'],'member':m.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'fields':fields});found.add(m.name)
  while p.stdout.read(1048576):pass
  err=p.stderr.read().decode(errors='replace');rc=p.wait()
  if rc or found!=set(targets):raise ValueError('incomplete '+str(rc)+' '+err[:256])
 except Exception as ex:
  errors.append({'object':rec['object'],'error':repr(ex)})
  if p.poll() is None:p.kill();p.wait()
(w/'classified-schema-matches.json').write_text(json.dumps({'rows':rows,'errors':errors},indent=2)+'\n')
from collections import Counter
print('matches',len(rows),'errors',len(errors),'phases',dict(Counter(r['fields'].get('phase','missing') for r in rows)))
for r in rows:
 if 'bridge' in r['fields'].get('phase',''):print(r)
