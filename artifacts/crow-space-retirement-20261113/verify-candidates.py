import collections,concurrent.futures,hashlib,json,os,subprocess,sys,threading,time,urllib.parse,urllib.request
from pathlib import Path
root=Path('/Users/n/Developer/numi-crow-rl-20261107').resolve(); commit='dffeaed67be0aa341a2643a4f0049f86f59b5dc2';token=sys.stdin.readline().strip();assert token
candidates=json.loads(Path('/tmp/crow-space-candidates.json').read_text());tracked=set(subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0'));op=subprocess.run(['lsof','-nP','-Fn'],capture_output=True,text=True);assert op.returncode in (0,1);opened={x[1:] for x in op.stdout.splitlines() if x.startswith('n/')};selected=[]
for c in sorted(candidates,key=lambda x:x['bytes'],reverse=True):
 p=root/c['path'];st=p.lstat()
 if c['path'] in tracked or st.st_nlink!=1 or not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root) or str(p) in opened:continue
 if any(x in c['path'] for x in ('long-walk-failure','launch-leg-wing-factorial')):continue
 x=subprocess.run(['xattr',str(p)],capture_output=True,text=True)
 if x.returncode or x.stdout.strip():continue
 selected.append((c,p,st))
 if len(selected)==100:break
print('Candidates selected:',len(selected),'GiB',round(sum(x[2].st_size for x in selected)/1024**3,3),flush=True)
lock=threading.Lock();out=Path('/tmp/crow-space-verified-batch1.jsonl');out.write_text('');counter=0
def work(item):
 global counter
 c,p,st=item;record={'path':c['path'],'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'mode':st.st_mode,'inode':st.st_ino,'device':st.st_dev,'git_blob':c['git_blob'],'commit':commit}
 try:
  sha=hashlib.sha256();git=hashlib.sha1(('blob %d\0'%st.st_size).encode())
  with p.open('rb') as f:
   while chunk:=f.read(1048576):sha.update(chunk);git.update(chunk)
  if git.hexdigest()!=c['git_blob']:raise ValueError('local blob differs from archived git tree')
  st2=p.stat();assert (st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)==(st2.st_size,st2.st_mtime_ns,st2.st_ino,st2.st_dev)
  url='https://raw.githubusercontent.com/Numi2/numi-lab/'+commit+'/'+urllib.parse.quote(c['path']);req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token});rh=hashlib.sha256();sz=0
  with urllib.request.urlopen(req,timeout=90) as response:
   while chunk:=response.read(1048576):rh.update(chunk);sz+=len(chunk)
  assert sz==st.st_size and rh.hexdigest()==sha.hexdigest(),'remote independently streamed bytes differ'
  record.update(sha256=sha.hexdigest(),status='verified')
 except Exception as exc: record.update(status='skipped',reason=str(exc))
 with lock:
  with out.open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n');f.flush()
  counter+=1
  if counter%10==0:print('Checked',counter,'of',len(selected),'last',record['status'],flush=True)
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:res=list(pool.map(work,selected))
print('Finished:',collections.Counter(x['status'] for x in res),'verified GiB',round(sum(x['bytes'] for x in res if x['status']=='verified')/1024**3,3),flush=True)
