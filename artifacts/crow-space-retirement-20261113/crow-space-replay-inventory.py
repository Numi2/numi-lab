from pathlib import Path
import json,gzip
r=Path('/Users/n/Developer/numi-crow-rl-20261107');tree=Path('/tmp/crow-space-archive-tree.txt');paths={s.split('\t',1)[1]:s.split()[2] for s in tree.read_text().splitlines() if '\t' in s};old=set()
for n in ('crow-space-candidates.json','crow-space-extra-candidates.json'):
 old.update(x['path'] for x in json.loads(Path('/tmp',n).read_text()))
out=[];errors=[]
for p in (r/'artifacts').rglob('*.json.gz'):
 if p.is_symlink() or not p.is_file():continue
 rel=str(p.relative_to(r));sz=p.stat().st_size
 if sz<5242880 or rel not in paths or rel in old:continue
 try:
  with gzip.open(p,'rb') as f:header=f.read(65536)
  if b'"body_names"' not in header or b'"action_fingerprint"' not in header:continue
 except Exception as e:errors.append([rel,str(e)]);continue
 out.append({'path':rel,'bytes':sz,'git_blob':paths[rel],'category':'structurally identified old replay'})
Path('/tmp/crow-space-more-replay-candidates.json').write_text(json.dumps(out));print('Structurally identified archived old replays:',len(out),'GiB',round(sum(x['bytes'] for x in out)/1024**3,3),'inspection errors',len(errors))
