from pathlib import Path
import json,hashlib
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-integrated-globalization-contact-memory-259-003';b=q/'build-source259-integrated-003';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='663a2bf205f2439200077d997804c429c33635d8606462f7a3f700a2d29a2743'
bm=json.loads((b/'source-build.json').read_text());assert bm['success'];bh=sha(b/'source-build.json')
t=(q/'run_native607_best_of_default_off_001.py').read_text()
for a,v in [
 ('source-best-of-cgls-255-001',s.name),('build-source255-best-of-005',b.name),
 ('best-of255-default-off-seven-phase-607','integrated259-default-off-seven-phase-611'),('native607-command-001','native611-command-001'),('native607-command.v1','native611-command.v1'),
 ('1efd6df0382dfe74e45d453fb22a81848d1fb251e99c2ab5cf6a8b8a5b062fc4','663a2bf205f2439200077d997804c429c33635d8606462f7a3f700a2d29a2743'),
 ('9d80b01f7a3910d64daa9a8a7f05cc1a54553d92f66e674d21a409d5852b7567',bh),
 ('Does source255 with best-of disabled preserve the complete accepted tiny seven-phase production history against canonical603?','Does integrated259-003 with best-of disabled preserve all32 files of the complete tiny seven-phase regression?'),
 ('Require completed seven-control history and all32 production files byte-identical to603. Only the default-off source255 selection path is added relative to its253 base;603 already exactly matches599. No mesh/time convergence or full-coupon qualification follows.','Require all seven controls and32 production files byte-identical to607/603. Integrated source contains the263 missing-registration repair and257/258 observer/memory changes, and retains the420b4f86 Human update. No convergence or full coupon claim follows.'),
 ("str(q/'analysis/native606-best-of-selection-command-001/receipt.json')","str(q/'analysis/source259-build-003/report.json')")]:
 assert a in t,a;t=t.replace(a,v)
p=q/'run_native611_integrated259_default_off_001.py';assert not p.exists();p.write_text(t)
u=t.replace('integrated259-default-off-seven-phase-611','integrated259-best-of-seven-phase-612').replace('native611-command','native612-command').replace("env=dict(old['environment'])","env=dict(old['environment']);env['NM_FEM_BASE_CGLS_BEST_OF']='1'")
u=u.replace('Does integrated259-003 with best-of disabled preserve all32 files of the complete tiny seven-phase regression?','Does integrated259-003 best-of complete seven phases with actual tested-alpha application and final production residual certification?').replace('Require all seven controls and32 production files byte-identical to607/603. Integrated source contains the263 missing-registration repair and257/258 observer/memory changes, and retains the420b4f86 Human update. No convergence or full coupon claim follows.','Require seven accepted controls and production certificates, inspect each selection and exact alpha, and compare the final last-selected-iteration rebuilt residual audit. Native608 failed before first accepted step because registrations were absent; source263 corrects only that host setup. No full coupon or convergence claim follows.')
p=q/'run_native612_integrated259_best_of_001.py';assert not p.exists();p.write_text(u)
# Full fresh schedule from585; no checkpoint state is transferred to a new program.
v=t.replace('integrated259-default-off-seven-phase-611','integrated259-best-of-strict-full-coupon-609').replace('native611-command','native609-command')
v=v.replace("q/'solver/source210-periodic-observer-off-544/prediction.json'","q/'solver/strict-state-source240-newton128-full-coupon-585/prediction.json'")
v=v.replace("env=dict(old['environment'])","env=dict(old['environment']);env['NM_FEM_BASE_CGLS_BEST_OF']='1'")
v=v.replace('Does integrated259-003 with best-of disabled preserve all32 files of the complete tiny seven-phase regression?','Can native best-of globalization pass the original strict control19 stall and complete the unchanged60-control .375degree coupon schedule?')
v=v.replace('Require all seven controls and32 production files byte-identical to607/603. Integrated source contains the263 missing-registration repair and257/258 observer/memory changes, and retains the420b4f86 Human update. No convergence or full coupon claim follows.','Fresh initial state, no cross-program checkpoint resume. Keep585 recipe, tools,8/2/1 mesh,dt5e-5,R5e-7,128Newton,FGMRES1024/restart16,local32/strict projection,mu1e-3/64CGLS and60-control loading/hold/unload/relax/release/post-release duration exactly. The sole operational change is BEST_OF=1. A healthy accepted Newton direction is compared with a separately tested CGLS direction from the same candidate/history, using the strictly lower recomputed residual and exactly its tested alpha. Full production certificate and rollback remain authoritative. Failure/timeouts remain failures; accepted prefixes do not qualify full history. This is still a bounded .375degree solver fixture, not90degree scoring, mesh/time convergence or assembly.')
v=v.replace("'--timeout','900'","'--timeout','7200'").replace('p.wait(timeout=960)','p.wait(timeout=7260)')
p=q/'run_native609_integrated259_best_of_strict_full_001.py';assert not p.exists();p.write_text(v)
print(json.dumps({'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':bh,'scripts':['run_native611_integrated259_default_off_001.py','run_native612_integrated259_best_of_001.py','run_native609_integrated259_best_of_strict_full_001.py']}))
