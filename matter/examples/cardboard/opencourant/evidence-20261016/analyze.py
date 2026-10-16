#!/usr/bin/env python3
"""Verify and summarize retained admission attempts; never promote exits to physics."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
adapters={sha(p):p.name for p in ROOT.glob('executed-adapter*.py')}
for path in sorted((ROOT/'runs').glob('*/run.json')):
 r=json.loads(path.read_text());d=path.parent
 changed=[n for n,v in r['outputs'].items() if not (d/n).is_file() or sha(d/n)!=v['sha256']]
 if changed: raise SystemExit(f'{d.name}: altered output {changed}')
 if r['adapter_sha256'] not in adapters: raise SystemExit(f'{d.name}: missing executed adapter')
 log='\n'.join(p.read_text(errors='replace') for p in d.glob('*.stdout'))
 if 'MUMPS required' in log: status='blocked_packaged_implicit_missing_MUMPS'
 elif 'Segmentation Violation' in log:status='failed_cycle_zero_crash'
 elif r['execution_complete']:status='runtime_execution_only_not_mechanics_qualified'
 elif not any('engine_' in ' '.join(c['argv']) for c in r['commands']):status='invalid_fixture_input'
 else:status='invalid_engine_input'
 peak=[]
 for p in d.glob('*.stderr'):
  m=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',p.read_text())
  if m:peak.append(int(m[1])*1024)
 rows.append({'attempt':d.name,'case':r['case_id'],'status':status,
              'elapsed_s':r['elapsed_s'],'max_solver_rss_bytes':max(peak) if peak else None,
              'stage_exit_codes':[c['exit_code'] for c in r['commands']],
              'receipt_sha256':sha(path),'output_identity_verified':True,
              'executed_adapter':adapters[r['adapter_sha256']]})
result={'schema':'cardboard.opencourant.admission-result.v1','admitted':False,
 'reason':'Packaged ARM64 runtime stops at cycle zero with MUMPS required in nonlinear implicit LAW1 and LAW117 controls, despite explicit PCG selection.',
 'release':'latest-20261011','source_commit':'cb760c206e346eff12161a2268b134ef9d48a4ef',
 'stopping_rule_applied':True,'attempt_count':len(rows),'attempts':rows,
 'qualification':{'backend_correctness':'NOT_ADMITTED','numerical_convergence':'NOT_RUN',
 'component_calibration':'NOT_COMPLETE','analogue_validation':'NOT_RUN',
 'physical_assembly':'NOT_RUN','crush_impact_prediction':'NOT_RUN'},
 'forecast':{'full_campaign_elapsed_time_s':None,'credible_shipping_resource_estimate':False,
 'reason':'No executable implicit slow-stage configuration admitted; single-element explicit timings cannot predict the shipping campaign.'},
 'scope_limits':['No score/fold strip, activation, restart-equivalence, or shipping segment run after the indispensable failure.',
 'Three explicit runtime completions are not mechanics qualification.',
 'Adhesion-02 CSV lacks titles; reaction channels are impulses, not calibrated force histories.',
 'Final exporter kinematic/output corrections after the retained probes remain unexecuted.',
 'Native Matter discrepancy remains frozen and unresolved.',
 'Paper and adhesion parser probes were interleaved; this was instrument repair, not confirmation in the listed fixture order.']}
(ROOT/'admission-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'attempts':len(rows),'output_hashes_verified':True,'admitted':False}))
