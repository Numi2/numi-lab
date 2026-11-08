#!/usr/bin/env python3
import io,json,os,subprocess,tarfile
from pathlib import Path
p=Path(__file__).parent;env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0');root='/Users/numi/numi-lab-policy-repair-main-20261108'
archives=json.loads((p/'surgical-archive-members.json').read_text())['archives'];out=[]
for a in archives:
 result={'object':a['object'],'path':a['paths'][0],'documents':[],'errors':[]}
 try:
  r=subprocess.run(['git','-C',root,'cat-file','blob',a['object']],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=True)
  with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as t:
   for m in t:
    name=m.name.removeprefix('./')
    if name not in ('receipt.log','checkpoint-precheck.log','checkpoint-postcheck.log','checkpoint-inputs.sha256'):continue
    if not m.isfile() or m.size>65536:raise ValueError('invalid bounded receipt '+name)
    result['documents'].append({'name':name,'text':t.extractfile(m).read().decode('utf-8','replace')})
 except Exception as e:result['errors'].append(repr(e))
 out.append(result)
print(json.dumps({'archives':out},indent=2))
