#!/usr/bin/env python3
import hashlib,json,os,re,subprocess,tarfile
from pathlib import Path
p=Path(__file__).parent;root='/Users/numi/numi-lab-policy-repair-main-20261108';env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0')
all=json.loads((p/'archive-inventory.json').read_text())['archive_blobs']; selected=[v for v in all if any(s.startswith('numi/evidence/fresh-robot-approach-20261108/') for s in v['paths'])]
assert len(selected)==18
results=[]
for v in selected:
 entry={'object':v['object'],'bytes':v['bytes'],'paths':v['paths'],'members':[],'errors':[]}
 try:
  r=subprocess.run(['git','-C',root,'cat-file','blob',v['object']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=30,check=True)
  assert len(r.stdout)==v['bytes'];entry['sha256']=hashlib.sha256(r.stdout).hexdigest()
  import io
  with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as t:
   for i,m in enumerate(t):
    if i>=100000:raise ValueError('archive member limit')
    if re.search(r'checkpoint|state|receiver|bridge|suture|needle|bite|manifest|receipt',m.name,re.I):entry['members'].append({'name':m.name,'size':m.size,'type':m.type.decode('latin1')})
   entry['member_count']=i+1 if 'i' in locals() else 0
 except Exception as e:entry['errors'].append(repr(e))
 results.append(entry)
print(json.dumps({'archive_count':len(selected),'archives':results},indent=2))
