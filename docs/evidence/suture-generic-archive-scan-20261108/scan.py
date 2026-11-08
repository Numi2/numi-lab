#!/usr/bin/env python3
import hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path
p=Path(__file__).parent
root='/Users/numi/numi-lab-reconciliation-20261108'
env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0')
all=json.loads((p/'inventory.json').read_text())['archive_blobs']
items=[v for v in all if not any(s.startswith('numi/evidence/fresh-robot-approach-20261108/') for s in v['paths'])]
items.sort(key=lambda v:(v['bytes'],v['object']))
assert len(items)==282
output=p/'results.jsonl'
existing={json.loads(x)['object'] for x in output.read_text().splitlines()} if output.exists() else set()
limit=int(sys.argv[1]) if len(sys.argv)>1 else len(items)
class Reader:
 def __init__(self,wrapped):self.w=wrapped;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.w.read(n);self.h.update(b);self.n+=len(b);return b
pattern=re.compile(r'tissue|suture|stitch|needle|receiver|bridge|first.?bite',re.I)
for v in [x for x in items if x['object'] not in existing][:limit]:
 entry={'object':v['object'],'paths':v['paths'],'expected_bytes':v['bytes'],'complete':False,'matches':[],'errors':[],'member_count':0}
 proc=None
 try:
  if not v['available']:raise ValueError('blob unavailable in inventory')
  if any(s.endswith(('.tar.zst','.tzst')) for s in v['paths']):raise ValueError('zstd archive not inspected by this scanner')
  proc=subprocess.Popen(['git','-C',root,'cat-file','blob',v['object']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  r=Reader(proc.stdout)
  with tarfile.open(fileobj=r,mode='r|*') as t:
   for m in t:
    entry['member_count']+=1
    if entry['member_count']>1000000:raise ValueError('member count limit')
    if pattern.search(m.name):
     if len(entry['matches'])>=10000:raise ValueError('matching member limit')
     entry['matches'].append({'name':m.name,'size':m.size,'type':m.type.decode('latin1')})
  while r.read(1024*1024):pass
  err=proc.stderr.read().decode('utf-8','replace');rc=proc.wait()
  entry.update({'sha256':r.h.hexdigest(),'bytes_read':r.n,'git_exit':rc})
  if rc:raise ValueError('git cat-file failed: '+err[:1024])
  if r.n!=v['bytes']:raise ValueError('compressed size mismatch')
  entry['complete']=True
 except Exception as e:
  entry['errors'].append(repr(e))
  if proc is not None and proc.poll() is None:proc.kill();proc.wait()
 with output.open('a') as f:f.write(json.dumps(entry,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 print(entry['object'],entry['complete'],entry['member_count'],len(entry['matches']),flush=True)
