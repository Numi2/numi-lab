#!/usr/bin/env python3
import json,os,re,subprocess
from pathlib import Path
p=Path(__file__).parent;root='/Users/numi/numi-lab-policy-repair-main-20261108';env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0');matches={};failed=[]
pattern=re.compile(r'\.(?:tar|tar\.gz|tgz|zip|tar\.xz|tar\.zst|tzst)$',re.I)
for line in (p/'remote-heads.txt').read_text().splitlines():
 sha,ref=line.split();r=subprocess.run(['git','-C',root,'ls-tree','-r','-z',sha],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
 if r.returncode:failed.append({'commit':sha,'ref':ref,'error':r.stderr.decode(errors='replace')[:500]});continue
 for row in r.stdout.split(b'\0'):
  if not row:continue
  meta,path=row.split(b'\t',1);name=path.decode('utf-8','surrogateescape')
  if not pattern.search(name):continue
  mode,kind,oid=meta.decode().split();entry=matches.setdefault(oid,{'object':oid,'paths':set(),'refs':set()});entry['paths'].add(name);entry['refs'].add(ref)
ids=sorted(matches);proc=subprocess.run(['git','-C',root,'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input=('\n'.join(ids)+'\n').encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
assert proc.returncode==0
results=proc.stdout.decode().splitlines();assert len(results)==len(ids)
for oid,line in zip(ids,results):
 parts=line.split();entry=matches[oid];entry['paths']=sorted(entry['paths']);entry['refs']=sorted(entry['refs']);entry['available']=len(parts)==3 and parts[1]=='blob';entry['bytes']=int(parts[2]) if entry['available'] else None
print(json.dumps({'heads':len((p/'remote-heads.txt').read_text().splitlines()),'failed_trees':failed,'archive_blobs':[matches[x] for x in ids]},indent=2))
