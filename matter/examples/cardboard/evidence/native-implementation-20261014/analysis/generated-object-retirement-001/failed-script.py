from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path('/Users/home/numi-cardboard-20261006');B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/generated-object-retirement-001';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];before=shutil.disk_usage(B).free
for n in range(90,109):
 D=R/f'build-cardboard-provenance-{n:03d}';P=D/'source-build.json'
 if not P.exists():continue
 m=json.loads(P.read_text());assert Path(m['build_dir'])==D
 protected={Path(p).resolve() for p in m['artifacts_sha256']}|{(D/p).resolve() for p in m.get('build_metadata_sha256',{})}
 files=[p for p in D.rglob('*.o') if p.is_file() and not p.is_symlink() and 'CMakeFiles' in p.parts];assert not protected.intersection(p.resolve() for p in files)
 for p,h in m['artifacts_sha256'].items():assert sha(Path(p))==h,p
 # The retained executables/libraries/metallibs, source receipts and logs remain.
 for p in files:
  l=subprocess.run(['lsof','-t','--',str(p)],capture_output=True,text=True);assert l.returncode==1 and not l.stdout,(p,l.stdout)
  rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),source_build_sha256=sha(P)));p.unlink()
(O/'receipt.json').write_text(json.dumps(dict(schema='numi.cardboard.generated-object-retirement.v1',files=rows,free_bytes_before=before,free_bytes_after=shutil.disk_usage(B).free,retained='All source snapshots, inputs, histories, executables, libraries, metallibs, source-build receipts, metadata and logs. Only regenerable CMake object files with no open handles were removed.',physical_evidence_removed=False),indent=2)+'\n');shutil.copy2(__file__,O/'retire.py');print('removed objects',len(rows),'bytes',sum(x['bytes'] for x in rows),'free',shutil.disk_usage(B).free,flush=True)
