from pathlib import Path
import hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');O=B/'analysis/source122-123-build-bindings-001';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[]
for name in ['122','123','123a']:
 d=R/f'build-cardboard-provenance-{name}';p=d/'source-build.json';j=json.loads(p.read_text());errors=[]
 for key in ['artifacts_sha256','build_metadata_sha256']:
  for rel,h in j.get(key,{}).items():
   a=Path(rel) if Path(rel).is_absolute() else d/rel
   if not a.is_file() or sha(a)!=h:errors.append(str(a))
 assert not errors,(name,errors)
 for rel,h in j['source_sha256'].items():assert sha(Path(j['source_snapshot'])/rel)==h,rel
 target=O/f'source-build-{name}.json';shutil.copy2(p,target)
 for q in ['configure.log','build.log','request.json','configure-command.json','build-command.json']:
  if (d/q).is_file():shutil.copy2(d/q,O/f'{name}-{q}')
 rows.append(dict(build=str(d),receipt_sha256=sha(p),success=j['success'],targets=j.get('targets'),source_inputs_and_all_advertised_artifacts_and_build_metadata_verified=True))
(O/'report.json').write_text(json.dumps(dict(rows=rows,limits='Build123 failed due to nonexistent requested target;123a uses correctprobe target and succeeds. No failure is converted into success or qualification.'),indent=2)+'\n');(O/'seal.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n');print('sealed122123builds')
