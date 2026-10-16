#!/usr/bin/env python3
from pathlib import Path
import hashlib, json
ROOT=Path('/Users/home/cardboard-implementation-20261014/analysis/native468-independent-audit-001')
SRC=Path('/Users/home/cardboard-implementation-20261014/analysis/native468-complete-failed-attempt-002')
RUN=Path('/Users/home/cardboard-implementation-20261014/solver/cgls-dt-half-cycle-468')
GUARD=Path('/Users/home/cardboard-implementation-20261014/analysis/native468-launch-guard-001/receipt.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
files={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name not in {'manifest.json','manifest.sha256'}}
manifest={
 'schema':'numi.cardboard.native468_independent_audit_manifest.v1',
 'audit_report_sha256':sha(ROOT/'report.json'),
 'source_analysis_inputs':{str(SRC/n):sha(SRC/n) for n in ('report.json','manifest.json','analyze.py','README.md')},
 'run_inputs':{str(RUN/n):sha(RUN/n) for n in ('prediction.json','receipt.json','output/result.json','output/observations.csv','stderr.log')},
 'capacity_guard_sha256':sha(GUARD),
 'files':files,
 'physical_qualification':False,
}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
(ROOT/'manifest.sha256').write_text(sha(ROOT/'manifest.json')+'  manifest.json\n')
print(json.dumps({'manifest_sha256':sha(ROOT/'manifest.json'),'report_sha256':sha(ROOT/'report.json'),'files':files},indent=2,sort_keys=True))
