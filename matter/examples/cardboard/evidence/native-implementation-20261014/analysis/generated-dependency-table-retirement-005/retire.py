from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path('/Users/home/numi-cardboard-20261006');B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/generated-dependency-table-retirement-005';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before=shutil.disk_usage(B).free;rows=[]
for D in [R/f'build-cardboard-provenance-{n:03d}' for n in range(1,118)]+[R/'build-cardboard-provenance-118-full',R/'build-cardboard-provenance-131']:
 P=D/'source-build.json'
 if not P.exists():continue
 m=json.loads(P.read_text())
 if not m.get('success') or 'artifacts_sha256' not in m:continue
 assert Path(m['build_dir'])==D
 for p,h in m['artifacts_sha256'].items():assert sha(Path(p))==h,p
 for p,h in m.get('build_metadata_sha256',{}).items():assert sha(D/p)==h,p
 protected={Path(p).resolve() for p in m['artifacts_sha256']}|{(D/p).resolve() for p in m.get('build_metadata_sha256',{})}
 files=[p for p in D.rglob('compiler_depend.*') if p.is_file() and not p.is_symlink() and 'CMakeFiles' in p.parts and p.name in ['compiler_depend.internal','compiler_depend.make']];assert not protected.intersection(p.resolve() for p in files)
 if files:
  l=subprocess.run(['lsof','-t','--',*[str(p) for p in files]],capture_output=True,text=True);assert l.returncode==1 and not l.stdout,(D,l.stdout)
 for p in files:rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),source_build_sha256=sha(P)))
(O/'preflight.json').write_text(json.dumps(dict(files=rows,free_bytes_before=before,retained='All source snapshots, inputs, histories, executables, libraries, metallibs, build receipts, CMake cache, compile commands and logs. Only closed generated compiler dependency tables are removed.'),indent=2)+'\n')
for r in rows:p=Path(r['path']);assert sha(p)==r['sha256'];p.unlink()
(O/'receipt.json').write_text(json.dumps(dict(schema='numi.cardboard.generated-dependency-table-retirement.v1',preflight_sha256=sha(O/'preflight.json'),files_removed=len(rows),bytes_removed=sum(r['bytes'] for r in rows),free_bytes_before=before,free_bytes_after=shutil.disk_usage(B).free,physical_evidence_removed=False),indent=2)+'\n');shutil.copy2(__file__,O/'retire.py');print((O/'receipt.json').read_text(),flush=True)
