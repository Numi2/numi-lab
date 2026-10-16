from pathlib import Path
import hashlib,json,csv,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');a=q/'solver/source210-periodic-observer-off-544/output';b=q/'solver/composed241-seven-phase-regression-588/output';d=q/'analysis/native588-composed-production-parity-001';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text());assert ra['status']==rb['status']=='completed' and ra['accepted_steps']==rb['accepted_steps']==7
names=sorted(p.name for p in a.iterdir() if p.is_file());assert names==sorted(p.name for p in b.iterdir() if p.is_file());records=[{'path':n,'original_sha256':sha(a/n),'composed_sha256':sha(b/n),'byte_identical':(a/n).read_bytes()==(b/n).read_bytes()} for n in names]
diff=[r for r in records if not r['byte_identical']]
assert [r['path'] for r in diff]==['manifest.json'],diff
old_manifest=json.loads((a/'manifest.json').read_text())
new_manifest=json.loads((b/'manifest.json').read_text())
policy=new_manifest['compensated_source_paper_projection']
assert policy.pop('strict_state_certification') is False
assert policy.pop('strict_scaled_correction_threshold') is None
assert new_manifest==old_manifest
physical_records=[x for x in records if x['path']!='manifest.json']
assert all(x['byte_identical'] for x in physical_records)

r={'schema':'numi.cardboard.native588-composed-production-parity.v1','verdict':'PASS_EXACT_COMPLETE_TINY_TRAJECTORY','parent_snapshot':'source210','composed_snapshot':'source241','source_frozen_sha256':'f64e6c5b12a0d69b944e002eb4c499533ac9b6cd6b852a96d869ce053c41c1db','source_build_sha256':'4255148cd4d29d8a0a2dbb5048a32cdd8a5c6a04a4d3e74a3b70a757638969fc','complete_controls':7,'max_bend_angle_degrees':.01,'files':records,'all_physical_production_files_byte_identical':True,'manifest_only_adds_explicit_disabled_strict_policy':True,'physical_files':physical_records,'limits':'Full loading, hold, unload, relaxation, release and post-release fixture comparison, .01degree only. Target60control coupon, all mesh/time convergence, source-grade damage/debonding and shipping assembly remain unqualified.','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'verdict':r['verdict'],'files':len(records),'report_sha256':sha(d/'report.json')}))
