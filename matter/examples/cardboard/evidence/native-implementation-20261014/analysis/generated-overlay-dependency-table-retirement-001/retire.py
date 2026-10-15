from pathlib import Path
import hashlib,json,subprocess,shutil,gzip
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/generated-overlay-dependency-table-retirement-001';O.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['mortar-union-four-patch-overlay-002','mortar-union-four-patch-overlay-003','mortar-union-four-patch-overlay-004','mortar-union-four-patch-overlay-005','private-mortar-ccd-314-overlay-001','private-mortar-witness-diagnostic-318-overlay-001','private-mortar-fix-overlay-320-001']
rows=[];protected={}
for name in names:
 root=B/'analysis'/name;d=root/'build-host'
 if not d.is_dir():continue
 m=root/'manifest.json';assert m.is_file()
 text=m.read_text();assert 'compiler_depend.make' not in text and 'compiler_depend.internal' not in text
 protected[str(m)]=sha(m)
 q=subprocess.run(['lsof','-Fn','+D',str(d)],capture_output=True,text=True);assert q.returncode==1 and not q.stdout
 for p in d.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.name in ['compiler_depend.make','compiler_depend.internal'] and 'CMakeFiles' in p.parts:
   rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
  elif p.is_file() and (p.suffix in ['.a','.metallib','.json'] or p.name in ['CMakeCache.txt','compile_commands.json'] or (p.stat().st_mode&0o111)):
   protected[str(p)]=sha(p)
j=dict(scope='Only closed generated CMake compiler dependency tables from seven task-owned completed overlays; all sources, logs, manifests, executables, static libraries and metallibs preserved',files=rows,protected=protected,free_bytes_before=shutil.disk_usage(B).free)
(O/'preflight.json.gz').write_bytes(gzip.compress(json.dumps(j,indent=2).encode(),mtime=0))
for r in rows:p=Path(r['path']);assert sha(p)==r['sha256'];p.unlink()
assert all(sha(Path(p))==h for p,h in protected.items())
r=dict(files_removed=len(rows),logical_bytes_removed=sum(r['bytes'] for r in rows),free_bytes_before=j['free_bytes_before'],free_bytes_after=shutil.disk_usage(B).free,all_protected_artifacts_unchanged=True)
(O/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))