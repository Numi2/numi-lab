from pathlib import Path
import hashlib,json,subprocess,shutil,datetime
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S149=B/'snapshots/source-cgls-block-right-preconditioned-149';S154=B/'snapshots/source-cgls-all-controls-newton128-154';O=B/'analysis/source154-root-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((B/'analysis/source154-default64-native466-parity-001/report.json').read_text())['parity_pass']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='67ba06be7f48981dda142ea35eedd792d7cd2ca5'
a=json.loads((S149/'frozen-source.json').read_text())['source_sha256'];b=json.loads((S154/'frozen-source.json').read_text())['source_sha256'];assert a.keys()==b.keys()
for k,h in a.items():assert sha(R/k)==h,('root differs from149',k)
for k,h in b.items():assert sha(S154/k)==h,('snapshot154 changed',k)
changes=[k for k in a if a[k]!=b[k]];assert set(changes)=={'matter/src/runtime.mm','matter/tools/cardboard_probe.mm'}
O.mkdir(exist_ok=False)
patch=subprocess.check_output(['git','diff','--no-index','--',str(S149/'matter/src/runtime.mm'),str(S154/'matter/src/runtime.mm')],text=True) if False else None
for k in changes:shutil.copy2(S154/k,R/k)
for k,h in b.items():assert sha(R/k)==h,('integrated mismatch',k)
(O/'root.patch').write_bytes(subprocess.check_output(['git','diff','--','matter/src/runtime.mm','matter/tools/cardboard_probe.mm'],cwd=R))
report={'schema':'numi.cardboard.source154-root-integration.v1','integrated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_revision':'67ba06be7f48981dda142ea35eedd792d7cd2ca5','source_files_verified':len(b),'all_selected_root_source_hashes_equal_snapshot154':True,'changed_source_paths':changes,'source154_frozen_sha256':sha(S154/'frozen-source.json'),'source154_build_sha256':sha(R/'build-cardboard-provenance-154/source-build.json'),'native466_default64_parity_passed':True,'native466_parity_report_sha256':sha(B/'analysis/source154-default64-native466-parity-001/report.json'),'new128_allcontrol_behavior':'native467 running from immutable154; outcome not claimed','operator_and_acceptance_thresholds':'unchanged','build_topology_limit':'Source154 built standalone matter with three artifacts; source153 root build passed before the two probe CLI/help edits. Native466 measures default numeric-policy complete-attempt parity across these build topologies. Root full-project rebuild of154 is not claimed.','scope':'Exact opt-in all-control selector routing and probe outer Newton ceiling128; no complete coupon or convergence claim.'}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'integrate.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({'inputs':{str(p):sha(p) for p in [S149/'frozen-source.json',S154/'frozen-source.json',B/'analysis/source154-default64-native466-parity-001/report.json',B/'analysis/cgls-all-controls-review-independent-153-002/manifest.json',B/'analysis/source154-build-topology-erratum-001/manifest.json']},'files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}},indent=2)+'\n');print(json.dumps(report))
