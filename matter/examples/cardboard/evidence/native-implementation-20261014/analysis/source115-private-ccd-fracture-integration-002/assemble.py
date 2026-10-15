from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014'); R=Path('/Users/home/numi-cardboard-20261006')
OLD=B/'snapshots/source-explicit-material-budget-109'; NEW=B/'snapshots/source-private-ccd-fracture-115'; O=B/'analysis/source115-private-ccd-fracture-integration-002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((OLD/'frozen-source.json').read_text()); s=json.loads((OLD/'configure-support-source.json').read_text())
for k,h in {**a['source_sha256'],**s['files']}.items(): assert sha(OLD/k)==h,k
assert NEW.is_dir(); O.mkdir(exist_ok=False)
for k,h in {**a['source_sha256'],**s['files']}.items(): assert sha(NEW/k)==h,k
changes={}
P=B/'analysis/deterministic-private-ccd-status-overlay-004'
assert sha(P/'manifest.json')=='e733d3ea87bb441160bf7f46173112a304fc1a74904dbc3d8648d330e635fe9a'
m=json.loads((P/'manifest.json').read_text())
for row in m['files']:
    p=(P/'overlay'/row['path']) if row['role']=='base_modified' else ((NEW/row['path']) if row['role']=='unchanged_abi_or_target_dependency' else (P/row['path']))
    assert sha(p)==row['sha256'],str(p)
r=subprocess.run(['patch','-p1','--batch','--forward','-i',str(P/'overlay.patch')],cwd=NEW,capture_output=True,text=True)
(O/'private-ccd-patch.log').write_text(r.stdout+r.stderr); assert r.returncode==0
for row in m['files']:
    if row['role']=='base_modified':
        assert (NEW/row['path']).read_bytes()==(P/'overlay'/row['path']).read_bytes(),row['path']
        changes[row['path']]=sha(NEW/row['path'])
P=B/'analysis/fracture-plane-ownership-overlay-114-001'
assert sha(P/'manifest.json')=='6bccfd38ef37d49624a2ec24acb27c69e3135e1e520c8da7c565f4343a164a3b'
m=json.loads((P/'manifest.json').read_text())
for p,h in m['files'].items(): assert sha(P/p)==h,p
for p,h in m['base_files_sha256'].items(): assert sha(NEW/p)==h,p
r=subprocess.run(['patch','-p1','--batch','--forward','-i',str(P/'patch.diff')],cwd=NEW,capture_output=True,text=True)
(O/'fracture-patch.log').write_text(r.stdout+r.stderr); assert r.returncode==0
for p in m['base_files_sha256']:
    assert (NEW/p).read_bytes()==(P/'overlay'/p).read_bytes(),p
    changes[p]=sha(NEW/p)
selected={k:sha(NEW/k) for k in set(a['source_sha256'])|set(changes)}
(NEW/'frozen-source.json').write_text(json.dumps({**a,'base_snapshot':str(OLD),'scope':'Production private-CCD deterministic failure selection and shared-face crack-area ownership rejection. No geometric acceptance, material, tolerance, or iteration-limit changes.','source_sha256':selected},indent=2)+'\n')
extra={str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file() and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']}; assert extra==s['files']
(NEW/'configure-support-source.json').write_text(json.dumps({**s,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(NEW),selected_source_files=len(selected),support_files=len(extra),changes=changes),indent=2)+'\n'); shutil.copy2(__file__,O/'assemble.py')
cmd=[v.replace('source-explicit-material-budget-109','source-private-ccd-fracture-115').replace('provenance-109','provenance-115') for v in json.loads((B/'build-command-109.json').read_text())]
(B/'build-command-115.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build115-runner.log').open('x') as f: code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for k,h in {**selected,**extra}.items(): assert sha(NEW/k)==h,k
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-115/source-build.json')),indent=2)+'\n'); print('build115',code,flush=True); raise SystemExit(code)
