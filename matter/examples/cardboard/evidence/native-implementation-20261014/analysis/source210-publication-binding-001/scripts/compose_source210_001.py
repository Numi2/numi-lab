from pathlib import Path
import argparse,shutil,json,hashlib,datetime,difflib,subprocess
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--overlay',action='append',required=True);a=p.parse_args()
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-cohesive-quadrature-area-205';s=q/'snapshots'/a.output;d=q/'analysis'/(a.output+'-composition-001')
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
bf=json.loads((base/'frozen-source.json').read_text());assert sha(base/'frozen-source.json')=='c77675e7aedc7f14c93c8bd7ea9e0a1e18bf162a68cf7e4022c43c34c6d141f2'
for n,h in bf['source_sha256'].items():assert sha(base/n)==h,n
s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n in bf['source_sha256']:
 t=s/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,t)
paths=set(bf['source_sha256']);layers=[]
for index,name in enumerate(a.overlay):
 o=q/'snapshots'/name;of=json.loads((o/'frozen-source.json').read_text())
 for n,h in of['source_sha256'].items():assert sha(o/n)==h,(name,n)
 assert set(bf['source_sha256'])<=set(of['source_sha256'])
 changed=[n for n,h in of['source_sha256'].items() if bf['source_sha256'].get(n)!=h];merged=[];new=[]
 for n in sorted(changed):
  dst=s/n;dst.parent.mkdir(parents=True,exist_ok=True)
  if n not in paths:
   shutil.copy2(o/n,dst);new.append(n)
  elif sha(dst)==of['source_sha256'][n]:
   pass
  elif sha(dst)==bf['source_sha256'].get(n):
   shutil.copy2(o/n,dst)
  else:
   patch=''.join(difflib.unified_diff((base/n).read_text().splitlines(True),(o/n).read_text().splitlines(True),fromfile='a/'+n,tofile='b/'+n))
   patchfile=d/('layer-'+str(index)+'-'+Path(n).name+'.patch');patchfile.write_text(patch)
   r=subprocess.run(['git','apply','--check',str(patchfile)],cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   if r.returncode:raise RuntimeError((n,r.stderr.decode()))
   subprocess.run(['git','apply',str(patchfile)],cwd=s,check=True)
   merged.append(n)
  paths.add(n)
 layers.append({'snapshot':str(o),'frozen_sha256':sha(o/'frozen-source.json'),'changed_from_base':sorted(changed),'merged_with_prior_overlay':merged,'new':new})
mapping={n:sha(s/n) for n in sorted(paths)}
f=dict(bf);f.update({'source_sha256':mapping,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'composed_source210':{'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'layers':layers,'qualification':'Source composition only; all runtime qualifications require actual combined-source runs.'}})
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
diffs=[n for n in mapping if mapping[n]!=bf['source_sha256'].get(n)]
r={'snapshot':str(s),'frozen_source_sha256':sha(s/'frozen-source.json'),'source_count':len(mapping),'layers':layers,'changed_from_base':diffs,'gpu_executed':False}
(d/'composition.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
