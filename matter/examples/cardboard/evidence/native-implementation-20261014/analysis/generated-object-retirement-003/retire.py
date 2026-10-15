from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path('/Users/home/numi-cardboard-20261006'); B=Path('/Users/home/cardboard-implementation-20261014'); O=B/'analysis/generated-object-retirement-003';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]; before=shutil.disk_usage(B).free
for D in [R/f'build-cardboard-provenance-{n:03d}' for n in (109,112,115,116,117)]+[R/'build-cardboard-main-compatibility014',R/'build-cardboard-main-compatibility016']:
 P=D/'source-build.json'
 if not P.exists():continue
 m=json.loads(P.read_text());assert m.get('success') and Path(m['build_dir'])==D
 for p,h in m['artifacts_sha256'].items():assert sha(Path(p))==h,p
 for p,h in m.get('build_metadata_sha256',{}).items():assert sha(D/p)==h,p
 protected={Path(p).resolve() for p in m['artifacts_sha256']}|{(D/p).resolve() for p in m.get('build_metadata_sha256',{})}
 files=[p for p in D.rglob('*.o') if p.is_file() and not p.is_symlink() and 'CMakeFiles' in p.parts]
 assert not protected.intersection(p.resolve() for p in files)
 if files:
  result=subprocess.run(['lsof','-t','--',*[str(p) for p in files]],capture_output=True,text=True)
  assert result.returncode==1 and not result.stdout,(D,result.stdout)
 for p in files:rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),source_build_sha256=sha(P)))
(O/'preflight.json').write_text(json.dumps(dict(files=rows,free_bytes_before=before,retained='All source snapshots, inputs, histories, binaries, dylibs, metallibs, build metadata, receipts and logs. Only closed regenerable CMake object files are removed.'),indent=2)+'\n')
for row in rows:
 p=Path(row['path']);assert sha(p)==row['sha256'];p.unlink()
(O/'receipt.json').write_text(json.dumps(dict(schema='numi.cardboard.generated-object-retirement.v1',preflight_sha256=sha(O/'preflight.json'),files_removed=len(rows),bytes_removed=sum(r['bytes'] for r in rows),free_bytes_before=before,free_bytes_after=shutil.disk_usage(B).free,physical_evidence_removed=False),indent=2)+'\n')
shutil.copy2(__file__,O/'retire.py');print((O/'receipt.json').read_text(),flush=True)
