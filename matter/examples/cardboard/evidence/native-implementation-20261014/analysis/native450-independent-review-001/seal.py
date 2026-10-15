#!/usr/bin/env python3
"""Seal the independent native450 review without copying source/run payloads."""
import hashlib, json, pathlib
D=pathlib.Path(__file__).parent
B=pathlib.Path('/Users/home/cardboard-implementation-20261014')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def item(p):
 p=pathlib.Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
RUN=B/'solver/bounded-cgls64-exact-state-450'; OFF=B/'solver/bounded-cgls64-capture-off-parity-452'; BASE445=B/'solver/cgls-disabled-history-parity-445'; REF419=B/'solver/two-stage-first-failure-rich-fd-419'; SNAP142=B/'snapshots/source-bounded-native-cgls-mask-repair-142'; SNAP125=B/'snapshots/source-status5-vtccd-diagnostics-125'
refs={
 'run450_prediction':RUN/'prediction.json','run450_receipt':RUN/'receipt.json','run450_result':RUN/'output/result.json','run450_output_manifest':RUN/'output/manifest.json','run450_stdout':RUN/'stdout.log','run450_stderr':RUN/'stderr.log','run450_cgls_snapshot':RUN/'base-cgls.bin','run450_status5':RUN/'first-status5.bin','run450_compiled_pack':RUN/'output/compiled.nmatterpack',
 'run452_prediction':OFF/'prediction.json','run452_receipt':OFF/'receipt.json','run452_result':OFF/'output/result.json','run452_stdout':OFF/'stdout.log','run452_stderr':OFF/'stderr.log','run452_status5':OFF/'first-status5.bin','run452_compiled_pack':OFF/'output/compiled.nmatterpack',
 'run445_prediction':BASE445/'prediction.json','run445_stderr':BASE445/'stderr.log','run445_compiled_pack':BASE445/'output/compiled.nmatterpack',
 'run419_prediction':REF419/'prediction.json','run419_receipt':REF419/'receipt.json','run419_stderr':REF419/'stderr.log','run419_operator_capture':REF419/'operator-capture',
 'snapshot142_source_module':SNAP142/'matter/tools/cardboard_cached_operator_audit.py',
 'snapshot125_operator_module':SNAP125/'matter/tools/cardboard_cached_operator_audit.py',
 'root_candidate_audit':B/'analysis/native450-cgls-candidate-audit-001/report.json',
 'root_capture_parity':B/'analysis/native450-capture-off-parity-001/report.json',
 'root_adjudication':B/'analysis/native450-cgls-adjudication-001/report.json',
}
files={n:item(D/n) for n in ('audit.py','README.md','report.json','validation.log','seal.py')}
manifest={
 'schema':'numi.cardboard.native450_independent_review_manifest.v1',
 'verdict':'LIMITED_CGLS_TRIAL_AND_CAPTURE_PARITY_CONFIRMED; FULL_SOLVER_CONVERGENCE_NOT_QUALIFIED',
 'review_files':files,
 'bound_evidence':{k:item(v) for k,v in refs.items()},
 'noncopied_prediction_inventory_audits':{
  'run450':{'prediction_sha256':sha(RUN/'prediction.json'),'checked_count':800,'inventory_sha256':json.loads((D/'report.json').read_text())['run']['input_hash_audit_on']['inventory_sha256']},
  'run452':{'prediction_sha256':sha(OFF/'prediction.json'),'checked_count':801,'inventory_sha256':json.loads((D/'report.json').read_text())['run']['input_hash_audit_capture_off']['inventory_sha256']},
 },
 'limits':['No native execution or source mutation in this review.','Reference operator capture is separate run419; matching world/pack/root/Newton and RHS are bound, but full candidate-node byte identity is not established.','No full coupon, solver-convergence, or physical qualification.','N63 Armijo result is absent due the 64-record cap; the true-residual cycle rows do not reveal that branch.'],
}
mp=D/'manifest.json';mp.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
seal={'schema':'numi.cardboard.native450_independent_review_seal.v1','manifest':{'path':str(mp),'bytes':mp.stat().st_size,'sha256':sha(mp)},'review_files':files}
(D/'seal.json').write_text(json.dumps(seal,sort_keys=True,indent=2)+'\n')
print(json.dumps({'manifest':item(mp),'seal':item(D/'seal.json')},indent=2))
