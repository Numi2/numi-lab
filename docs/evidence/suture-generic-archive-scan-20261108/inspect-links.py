import json,subprocess,os,posixpath
from pathlib import Path
p=Path(__file__).parent; repo='/Users/numi/numi-lab-reconciliation-20261108'; env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0')
inventory=json.loads((p/'inventory.json').read_text())['archive_blobs'];byid={x['object']:x for x in inventory}
heads={r:h for h,r in (s.split() for s in Path('/Users/numi/numi-lab-reconciliation-20261108/docs/evidence/suture-branch-tip-search-20261108/remote-heads.txt').read_text().splitlines())}
scan={x['object']:x for x in map(json.loads,(p/'results.jsonl').read_text().splitlines())};out=[]
for z in scan.values():
 if z['complete'] or z['expected_bytes']!=91:continue
 original=byid[z['object']]; path=z['paths'][0];ref=next(r for r in original['refs'] if r in heads);commit=heads[ref]
 def tree(path):return subprocess.check_output(['git','-C',repo,'ls-tree',commit,'--',path],env=env,text=True).strip()
 mode=tree(path);target=subprocess.check_output(['git','-C',repo,'cat-file','blob',z['object']],env=env,text=True); resolved=posixpath.normpath(posixpath.join(posixpath.dirname(path),target));targetline=tree(resolved)
 obj=targetline.split()[2] if targetline else None
 out.append({'object':z['object'],'ref':ref,'commit':commit,'path':path,'ls_tree':mode,'target':target,'resolved':resolved,'target_ls_tree':targetline,'target_in_inventory':obj in byid,'target_scan_complete':scan.get(obj,{}).get('complete')})
(p/'link-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
