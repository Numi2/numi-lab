import hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path
w=Path(sys.argv[1]); root=Path(sys.argv[2]); limit=int(sys.argv[3]); env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0')
inventory=json.loads((root/'docs/evidence/suture-branch-tip-search-20261108/archive-inventory.json').read_text())['archive_blobs']
links={r['object'] for r in json.loads((root/'docs/evidence/suture-generic-archive-scan-20261108/link-analysis.json').read_text())}
items=sorted((r for r in inventory if r['object'] not in links),key=lambda r:(r['bytes'],r['object']))
out=w/'archive-content.jsonl';done={json.loads(line)['object'] for line in out.read_text().splitlines()} if out.exists() else set()
class Reader:
 def __init__(self,w):self.w=w;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.w.read(n);self.h.update(b);self.n+=len(b);return b
schema=re.compile(rb'(?m)^schema\tnumi\.dual-psm-suture-handoff-state\.v3\r?$')
phase=re.compile(rb'(?m)^phase\ttissue-receiver-bridge-start\r?$')
for item in [r for r in items if r['object'] not in done][:limit]:
 e={'object':item['object'],'paths':item['paths'],'expected_bytes':item['bytes'],'complete':False,'members':0,'regular_prefixes':0,'matches':[],'errors':[]}; g=z=None
 try:
  if not item['available']:raise ValueError('blob absent in frozen inventory')
  g=subprocess.Popen(['git','-C',str(root),'cat-file','blob',item['object']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  r=Reader(g.stdout)
  if any(p.endswith(('.tar.zst','.tzst')) for p in item['paths']):
   z=subprocess.Popen(['/opt/homebrew/bin/zstd','-q','-d','-c'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   # zstd members in this inventory contain no names resembling a checkpoint;
   # this bounded prefix pass still handles them using a second streaming process.
   import threading
   def feed():
    try:
     while True:
      b=r.read(1024*1024)
      if not b:break
      z.stdin.write(b)
    finally:z.stdin.close()
   feeder=threading.Thread(target=feed,daemon=True);feeder.start();stream=z.stdout;mode='r|'
  else:stream=r;mode='r|*'
  with tarfile.open(fileobj=stream,mode=mode) as tar:
   for m in tar:
    e['members']+=1
    if e['members']>1_000_000:raise ValueError('member count limit')
    if not m.isfile():continue
    f=tar.extractfile(m)
    if f is None:raise ValueError('regular member unreadable')
    prefix=f.read(min(m.size,65536));e['regular_prefixes']+=1
    a=bool(schema.search(prefix));b=bool(phase.search(prefix))
    if a or b:e['matches'].append({'name':m.name,'size':m.size,'prefix_bytes':len(prefix),'schema':a,'phase':b})
  if z:
   while z.stdout.read(1024*1024):pass
   feeder.join();ze=z.stderr.read().decode(errors='replace');zr=z.wait()
   if zr:raise ValueError('zstd exit '+str(zr)+' '+ze[:512])
  else:
   while r.read(1024*1024):pass
  err=g.stderr.read().decode(errors='replace');rc=g.wait()
  e.update(git_exit=rc,bytes_read=r.n,compressed_sha256=r.h.hexdigest())
  if rc or r.n!=item['bytes']:raise ValueError('source size/exit mismatch '+err[:512])
  e['complete']=True
 except Exception as ex:
  e['errors'].append(repr(ex))
  for proc in (z,g):
   if proc is not None and proc.poll() is None:proc.kill();proc.wait()
 with out.open('a') as f:f.write(json.dumps(e,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 print(e['object'],'complete',e['complete'],'members',e['members'],'prefixes',e['regular_prefixes'],'matches',len(e['matches']),flush=True)
