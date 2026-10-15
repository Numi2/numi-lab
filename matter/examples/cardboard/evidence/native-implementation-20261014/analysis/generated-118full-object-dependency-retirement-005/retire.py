from pathlib import Path
import datetime, hashlib, json, shutil, subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
T=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-118-full')
O=B/'analysis/generated-118full-object-dependency-retirement-005'
O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((B/'cpu-checks-118-full/receipt.json').read_text());assert receipt['exit_code']==0
build=json.loads((T/'source-build.json').read_text());assert build['success']
protected={Path(p):h for p,h in build['artifacts_sha256'].items()}
protected.update({T/p:h for p,h in build['build_metadata_sha256'].items()})
protected[T/'source-build.json']=sha(T/'source-build.json')
for p,h in protected.items():assert sha(p)==h,p
rows=[]
for p in sorted(T.rglob('*')):
 if not p.is_file() or p.is_symlink() or 'CMakeFiles' not in p.parts:continue
 if not (p.name.endswith('.o') or p.name.endswith('.o.d')):continue
 assert p not in protected
 rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),kind='regenerable closed CMake object/dependency'))
assert rows
q=subprocess.run(['lsof','-Fn','+D',str(T)],capture_output=True,text=True)
(O/'open-handles.log').write_text(q.stdout+q.stderr)
assert q.returncode==1 and not q.stdout.strip(), 'Build directory has open handles; retain everything'
free_before=shutil.disk_usage(T).free
pre=dict(schema='numi.cardboard.regenerable-build-retirement.v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),build=str(T),source_build_sha256=sha(T/'source-build.json'),verified_closed=True,protected={str(p):h for p,h in protected.items()},files=rows,free_bytes_before=free_before)
(O/'preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
for row in rows:
 p=Path(row['path']);assert sha(p)==row['sha256'];p.unlink()
for p,h in protected.items():assert sha(p)==h,p
(O/'result.json').write_text(json.dumps(dict(files_removed=len(rows),logical_bytes_removed=sum(x['bytes'] for x in rows),free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(T).free,all_compiled_artifacts_and_build_metadata_hashes_preserved=True,source_and_evidence_removed=False,preflight_sha256=sha(O/'preflight.json')),indent=2)+'\n')
(O/'retire.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in sorted(O.iterdir()) if p.is_file()},indent=2)+'\n')
print((O/'result.json').read_text())
