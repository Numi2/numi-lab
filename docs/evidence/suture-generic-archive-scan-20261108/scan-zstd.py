import hashlib,json,os,re,subprocess,tarfile
from pathlib import Path
p=Path(__file__).parent;root='/Users/numi/numi-lab-reconciliation-20261108';env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0')
items=[v for v in json.loads((p/'inventory.json').read_text())['archive_blobs'] if any(s.endswith(('.tar.zst','.tzst')) for s in v['paths'])];assert len(items)==3
pattern=re.compile(r'tissue|suture|stitch|needle|receiver|bridge|first.?bite',re.I);out=[]
for v in items:
 e={'object':v['object'],'paths':v['paths'],'expected_bytes':v['bytes'],'complete':False,'member_count':0,'matches':[],'errors':[]};g=z=None
 try:
  g=subprocess.Popen(['git','-C',root,'cat-file','blob',v['object']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  z=subprocess.Popen(['/opt/homebrew/bin/zstd','-q','-d','-c'],stdin=g.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE);g.stdout.close()
  with tarfile.open(fileobj=z.stdout,mode='r|') as t:
   for m in t:
    e['member_count']+=1
    if e['member_count']>1000000:raise ValueError('member count limit')
    if pattern.search(m.name):
     if len(e['matches'])>=10000:raise ValueError('matching member limit')
     e['matches'].append({'name':m.name,'size':m.size,'type':m.type.decode('latin1')})
  while z.stdout.read(1024*1024):pass
  ze=z.stderr.read().decode('utf8','replace');zr=z.wait();ge=g.stderr.read().decode('utf8','replace');gr=g.wait()
  e.update({'git_exit':gr,'zstd_exit':zr})
  if gr or zr:raise ValueError('pipeline failed: '+ge[:512]+ze[:512])
  h=hashlib.sha256();n=0
  a=subprocess.Popen(['git','-C',root,'cat-file','blob',v['object']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  while True:
   data=a.stdout.read(1024*1024)
   if not data:break
   n+=len(data);h.update(data)
  ae=a.stderr.read();ar=a.wait();e.update({'sha256':h.hexdigest(),'bytes_read':n})
  if ar or n!=v['bytes']:raise ValueError('hash read failed '+repr(ae))
  e['complete']=True
 except Exception as ex:
  e['errors'].append(repr(ex))
  for proc in (z,g):
   if proc is not None and proc.poll() is None:proc.kill();proc.wait()
 out.append(e);(p/'zstd-results.json').write_text(json.dumps(out,indent=2)+'\n');print(e['object'],e['complete'],e['member_count'],len(e['matches']),flush=True)
