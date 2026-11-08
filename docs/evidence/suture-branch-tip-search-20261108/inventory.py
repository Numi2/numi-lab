#!/usr/bin/env python3
import json,os,re,subprocess
from pathlib import Path
root=Path('/Users/numi/numi-lab-policy-repair-main-20261108');here=Path(__file__).parent
env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0')
heads=[line.split() for line in (here/'remote-heads.txt').read_text().splitlines()]
surgery=re.compile(r'suture|surgic|needle|stitch|sewing|opposing.bite|receiver.bridge|pull.complete',re.I)
artifact=re.compile(r'check.?point|state|archive|receipt|snapshot|manifest|\.tar(?:\.gz|\.zst)?$|\.zip$|\.bin$|\.npz$|\.safetensors$',re.I)
phase=re.compile(r'tissue.receiver.bridge|tissue.suture.pull.complete|tissue.receiver.acquire',re.I)
items={};failed=[];total=0
for sha,ref in heads:
 proc=subprocess.run(['git','-C',str(root),'ls-tree','-r','-z',sha],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
 if proc.returncode:
  failed.append({'ref':ref,'commit':sha,'exit':proc.returncode,'stderr':proc.stderr.decode(errors='replace')[:500]});continue
 total+=1
 for raw in proc.stdout.split(b'\0'):
  if not raw:continue
  meta,path=raw.split(b'\t',1);name=path.decode('utf-8','surrogateescape')
  if not (phase.search(name) or (surgery.search(name) and artifact.search(name))):continue
  mode,kind,oid=meta.decode().split();key=(name,oid);entry=items.setdefault(key,{'path':name,'object':oid,'type':kind,'mode':mode,'refs':[]});entry['refs'].append(ref)
print(json.dumps({'heads':len(heads),'searched_trees':total,'failed_trees':failed,'candidates':sorted(items.values(),key=lambda x:x['path'])},indent=2,ensure_ascii=True))
