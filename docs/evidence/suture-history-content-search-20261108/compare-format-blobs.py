import collections,json,os,re,subprocess,sys
from pathlib import Path
w=Path(sys.argv[1]);root=Path(sys.argv[2]);rx=re.compile(r'\.(?:tar|tgz|tar\.gz|tar\.zst|tzst|tar\.xz|txz|tar\.bz2|tbz2|zip|7z)$',re.I)
known=json.loads((root/'docs/evidence/suture-branch-tip-search-20261108/archive-inventory.json').read_text())
tip={row['object'] for row in known['archive_blobs']}; items=collections.defaultdict(set);commit=None
for line in (w/'format-history.txt').read_text().splitlines():
 if line.startswith('commit '):commit=line.split()[1]
 elif commit and rx.search(line):items[(commit,line)].add(line)
env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0'); outputs=[];errors=[]
for commit,path in sorted(items):
 p=subprocess.run(['git','ls-tree','-z',commit,'--',path],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 parts=p.stdout.rstrip(b'\0').split(b'\t',1)
 if p.returncode or len(parts)!=2:
  errors.append({'commit':commit,'path':path,'code':p.returncode,'stderr':p.stderr.decode(errors='replace')});continue
 meta=parts[0].decode().split();observed=parts[1].decode(errors='surrogateescape')
 if len(meta)!=3 or observed!=path:
  errors.append({'commit':commit,'path':path,'unexpected':p.stdout.hex()});continue
 outputs.append({'commit':commit,'path':path,'mode':meta[0],'kind':meta[1],'object':meta[2],'at_tip':meta[2] in tip})
new=[r for r in outputs if not r['at_tip']]
(w/'format-object-comparison.json').write_text(json.dumps({'pairs':len(items),'errors':errors,'rows':outputs,'historical_only':new},indent=2)+'\n')
print('history pairs',len(items),'resolved',len(outputs),'errors',len(errors),'historical-only rows',len(new),'unique objects',len({r['object'] for r in new}))
for row in new:print(row['commit'],row['object'],row['path'])
