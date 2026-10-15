from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-padded-trial-summary-120';N=B/'snapshots/source-valid-summary-tet858-121'
O=B/'analysis/source121-integration-001';A=B/'analysis/armijo-all-trial-summary-overlay-003';P=B/'analysis/source392-tet858-point-overlay-001';H=B/'analysis/projection-trace-host-capacity-121-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((S/'frozen-source.json').read_text());support=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**old['source_sha256'],**support['files']}.items():assert sha(S/p)==h,p
assert sha(P/'manifest.json')=='65d694c33e7266bb60fe327848f2d09c098ef09f34abace8cab67d8cdb1bc8e5'
assert sha(P/'source/cardboard_source392_tet_point_check.mm')=='ae8e3cbc64dbe4058562ee766ed10ce0d27f841daa4943e6baa7be28a0405900'
a=json.loads((A/'manifest.json').read_text())
assert sha(A/'manifest.json')=='e285f0471b68b61e9327cdec839c07957284c61764edeb728a901300d10fbf01'
for rel,h in a['files'].items():assert sha(A/rel)==h,rel
assert a['base']['frozen_source_runtime_sha256']==sha(S/'matter/src/runtime.mm')
hr=json.loads((H/'report.json').read_text());assert sha(S/hr['path'])==hr['base_sha256'];assert sha(H/'overlay'/hr['path'])==hr['overlay_sha256']
assert not N.exists() and not O.exists();O.mkdir()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True);(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
shutil.copy2(A/'overlay/matter/src/runtime.mm',N/'matter/src/runtime.mm')
shutil.copy2(H/'overlay'/hr['path'],N/hr['path'])
shutil.copy2(P/'source/cardboard_source392_tet_point_check.mm',N/'matter/tools/cardboard_source392_tet_point_check.mm')
cm=N/'matter/CMakeLists.txt';cm.write_text(cm.read_text()+'\n'+(P/'CMake-fragment.txt').read_text())
selected={p:sha(N/p) for p in old['source_sha256']};selected['matter/tools/cardboard_source392_tet_point_check.mm']=sha(N/'matter/tools/cardboard_source392_tet_point_check.mm');extra={p:sha(N/p) for p in support['files']}
(N/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(S),'scope':'Source120 plus readonly diagnostic manifest serializer repair overlay003 and exact retained source392/tet858 native point replay host, and repair C1 host trace capacity16to32 to match Metal11976wordwriter. No physical operators or acceptance thresholds changed.','source_sha256':selected},indent=2)+'\n')
(N/'configure-support-source.json').write_text(json.dumps({**support,'files':extra},indent=2)+'\n')
changes={p:{'source120':old['source_sha256'].get(p),'source121':h} for p,h in selected.items() if old['source_sha256'].get(p)!=h}
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),base_snapshot=str(S),runtime_overlay_manifest_sha256=sha(A/'manifest.json'),point_overlay_manifest_sha256=sha(P/'manifest.json'),trace_host_manifest_sha256=sha(H/'manifest.json'),changes=changes,source_sha256=selected,support_sha256=extra,physical_steps=0),indent=2)+'\n')
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(N/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(N),'--build-dir',str(R/'build-cardboard-provenance-121'),'--cmake-source-relative','.','--jobs','4']
for target in ['numi-matter-cardboard-full-contact-armijo-check','numi-matter-cardboard-source392-tet858-point-check','numi-matter-cardboard-asymmetric-c1-paper-gpu-check']:
 cmd+=['--target',target]
(B/'build-command-121.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build121-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(N/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-121/source-build.json')),indent=2)+'\n')
print('build121',code,flush=True);raise SystemExit(code)
