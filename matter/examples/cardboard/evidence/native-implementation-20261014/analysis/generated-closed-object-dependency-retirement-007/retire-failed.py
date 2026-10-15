from pathlib import Path
import datetime, hashlib, json, shutil, subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
O=B/'analysis/generated-closed-object-dependency-retirement-007';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for number in range(68,87):
 T=R/f'build-cardboard-provenance-{number:03d}'
 if not T.exists():continue
 build=json.loads((T/'source-build.json').read_text())
 protected={Path(p):h for p,h in build['artifacts_sha256'].items()}
 protected.update({T/p:h for p,h in build['build_metadata_sha256'].items()})
 protected[T/'source-build.json']=sha(T/'source-build.json')
 bad=[str(p) for p,h in protected.items() if not p.is_file() or sha(p)!=h]
 if bad:results.append(dict(build=str(T),status='retained-protected-artifact-mismatch',mismatches=bad));continue
 q=subprocess.run(['lsof','-Fn','+D',str(T)],capture_output=True,text=True)
 (O/f'open-handles-{number:03d}.log').write_text(q.stdout+q.stderr)
 if q.returncode!=1 or q.stdout.strip():results.append(dict(build=str(T),status='retained-open-handles'));continue
 rows=[]
 for p in sorted(T.rglob('*')):
  if not p.is_file() or p.is_symlink() or 'CMakeFiles' not in p.parts:continue
  if not (p.name.endswith('.o') or p.name.endswith('.o.d')):continue
  assert p not in protected
  rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),kind='regenerable closed CMake object/dependency'))
 if not rows:results.append(dict(build=str(T),status='already-absent'));continue
 before=shutil.disk_usage(T).free
 pre=dict(schema='numi.cardboard.regenerable-build-retirement.v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),build=str(T),verified_closed=True,protected={str(p):h for p,h in protected.items()},files=rows,free_bytes_before=before)
 pp=O/f'preflight-{number:03d}.json';pp.write_text(json.dumps(pre,indent=2)+'\n')
 for row in rows:
  p=Path(row['path']);assert sha(p)==row['sha256'];p.unlink()
 for p,h in protected.items():assert sha(p)==h,p
 results.append(dict(build=str(T),status='removed-generated-objects-only',files_removed=len(rows),logical_bytes_removed=sum(x['bytes'] for x in rows),free_bytes_before=before,free_bytes_after=shutil.disk_usage(T).free,all_compiled_artifacts_and_build_metadata_hashes_preserved=True,source_and_evidence_removed=False,preflight_sha256=sha(pp)))
(O/'result.json').write_text(json.dumps(dict(results=results,logical_bytes_removed=sum(x.get('logical_bytes_removed',0) for x in results),free_bytes_after=shutil.disk_usage(R).free),indent=2)+'\n')
(O/'retire.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in sorted(O.iterdir()) if p.is_file()},indent=2)+'\n')
print('retirement007',sum(x.get('logical_bytes_removed',0) for x in results),shutil.disk_usage(R).free)
