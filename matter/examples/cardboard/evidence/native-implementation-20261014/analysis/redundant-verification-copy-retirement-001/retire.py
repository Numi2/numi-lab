from pathlib import Path
import hashlib,json,shutil,subprocess,time
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
O=B/'analysis/redundant-verification-copy-retirement-001';O.mkdir(exist_ok=False)
rows=[];before=shutil.disk_usage(B).free
for n in [105,106,107]:
 P=B/f'analysis/retained-source-snapshot{n}-001';v=json.loads((P/'restore-cli-verification.json').read_text());assert v['exact_bytes_verified'] and v['actual_build_success']
 T=Path(v['output']);assert T==B/f'analysis/source{n}-restore-cli-verification-001'
 mf=json.loads((T/'frozen-source.json').read_text());S=Path(json.loads((R/f'build-cardboard-provenance-{n}/source-build.json').read_text())['source_snapshot'])
 d={**mf['source_sha256'],**json.loads((T/'configure-support-source.json').read_text())['files']}
 for k,h in d.items():assert sha(T/k)==sha(S/k)==h,k
 l=subprocess.run(['lsof','-nP','+D',str(T)],capture_output=True,text=True);assert l.returncode==1 and not l.stdout,(T,l.stdout,l.stderr)
 rows.append(dict(removed_redundant_verification_copy=str(T),exact_source_files_verified=len(d),original_snapshot_retained=str(S),restore_manifest_sha256=sha(P/'manifest.json'),prior_verification_sha256=sha(P/'restore-cli-verification.json'),open_handles=False))
 shutil.rmtree(T)
(O/'receipt.json').write_text(json.dumps(dict(schema='numi.cardboard.redundant-copy-retirement.v1',rows=rows,free_bytes_before=before,free_bytes_after=shutil.disk_usage(B).free,only_redundant_task_owned_verification_copies_removed=True),indent=2)+'\n');shutil.copy2(__file__,O/'retire.py');print((O/'receipt.json').read_text())
