from pathlib import Path
import hashlib,json,subprocess,shutil,gzip
B=Path('/Users/home/cardboard-implementation-20261014');T=B/'archive-unintended-preflight-recursion-20261015-001';O=B/'analysis/verified-recursive-duplicate-copy-retirement-001';O.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=subprocess.run(['lsof','-Fn','+D',str(T)],capture_output=True,text=True);(O/'open-handles.log').write_text(q.stdout+q.stderr);assert q.returncode==1 and not q.stdout
rows=[];retained=[]
for p in sorted(T.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 relative=p.relative_to(T);original=B/relative
 if not original.is_file() or original.is_symlink() or p.stat().st_size!=original.stat().st_size:
  retained.append(str(relative));continue
 h=sha(p)
 if sha(original)!=h:retained.append(str(relative));continue
 rows.append(dict(duplicate=str(p),canonical=str(original),bytes=p.stat().st_size,sha256=h))
j=dict(scope='Only byte-identical task-owned copies accidentally nested in an earlier preflight archive; canonical sources/evidence remain at original exact paths. Unmatched files remain.',files=rows,unmatched_retained=retained,free_bytes_before=shutil.disk_usage(B).free)
(O/'preflight.json.gz').write_bytes(gzip.compress(json.dumps(j,indent=2).encode(),mtime=0))
for r in rows:
 p=Path(r['duplicate']);original=Path(r['canonical']);assert sha(p)==sha(original)==r['sha256'];p.unlink()
for r in rows:assert sha(Path(r['canonical']))==r['sha256']
r=dict(files_removed=len(rows),logical_bytes_removed=sum(r['bytes'] for r in rows),unmatched_files_retained=len(retained),free_bytes_before=j['free_bytes_before'],free_bytes_after=shutil.disk_usage(B).free,all_canonical_bytes_reverified=True)
(O/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))