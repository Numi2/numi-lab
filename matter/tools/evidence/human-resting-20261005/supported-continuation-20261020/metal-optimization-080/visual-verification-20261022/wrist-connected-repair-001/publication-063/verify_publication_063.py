#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
P=Path(__file__).resolve().parent; payload=P/'payload'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
pins=json.loads((P/'source-input-pins-063.json').read_text())
assert pins['base_origin_main']=='f29a1f2d25d351423d03ef0954120eaae11f5517'
assert pins['native_or_physical_admission'] is False
pl=payload/'040'
fit5=json.loads((pl/'attempt-005/fit-001/fit-report.json').read_text())
fit6=json.loads((pl/'attempt-006/trust-constr-002/fit-report.json').read_text())
a6=json.loads((pl/'review/root-independent-candidate-006.json').read_text())
l8=json.loads((pl/'layer-depth/feasibility-008/report.json').read_text())
l9=json.loads((pl/'layer-depth/transition-009/report.json').read_text())
assert fit5['status']=='candidate_rejected_or_incomplete'
assert fit6['status']=='candidate_rejected_or_incomplete'
assert a6['nonpositive_cells']==4 and a6['quality_below_1e_5']==6
assert l8['nonpositive_tets']==1 and l8['route_constraint_violations_F32']==10
assert l9['nonpositive_tets']==0 and l9['q_below_1e_5']==0
assert l9['route_constraint_violations_F32']==10 and l9['O12_violations_F32']==12
cor=json.loads((pl/'route-interpretation-correction-063.json').read_text())
base=json.loads((pl/'review/baseline-route-report-010.json').read_text())
assert cor['baseline_route_audit']['report_sha256']==sha(pl/'review/baseline-route-report-010.json')
assert base['counts']['hit_segment_bank_pairs']==192 and base['counts']['proper_hits']==256
assert cor['candidate_005_006_008_postfit_exact_route_gate_included'] is False and cor['candidate_009_exact_route_audit_included'] is True
r9=json.loads((pl/'review/depth009-route-report-012.json').read_text())
assert r9['counts']['hit_segment_bank_pairs']==74 and r9['counts']['proper_hits']==76
assert r9['columns_with_hits']==73 and r9['construction_planes']==0 and r9['uncertainties']==0

r14=json.loads((pl/'baseline-depth/014/report.json').read_text())
r15=json.loads((pl/'baseline-depth/015-exact-routes.json').read_text())
assert r14['nonpositive_tets']==0 and r14['q_below_1e_5']==0 and r14['outer_nodes_moved']==0
assert r15['counts']['hit_segment_bank_pairs']==8 and r15['counts']['proper_hits']==8 and r15['uncertainties']==0
listed=json.loads((P/'payload-files-063.json').read_text()); expected={x['path']:x for x in listed['files']}
actual={}
for p in payload.rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts: continue
 rel=p.relative_to(payload).as_posix(); actual[rel]={'sha256':sha(p),'size_bytes':p.stat().st_size}
assert set(actual)==set(expected),(set(actual)-set(expected),set(expected)-set(actual))
for n,x in actual.items(): assert x=={'sha256':expected[n]['sha256'],'size_bytes':expected[n]['size_bytes']},n
print(json.dumps({'status':'PASS_CONTENT_AND_REJECTION_SEMANTICS','payload_files':len(actual),'payload_bytes':sum(x['size_bytes'] for x in actual.values()),'python':sys.version},sort_keys=True))
