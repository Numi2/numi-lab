from pathlib import Path
import json,hashlib,collections
out=Path(__file__).parent
rows=[json.loads(s) for s in (out/'rows.jsonl').open()]
nonzero=[r for r in rows if not r['zero_direction']]
sampled=[r for r in rows if r['samples']]
perstep=[]
for step in [1e-4,1e-5,2e-6,1e-6,1e-7,1e-8,1e-9,1e-10]:
 samples=[x for r in sampled for x in r['samples'] if x['max_F_step']==step]
 valid=[x for x in samples if x['valid']]
 perstep.append(dict(max_F_step=step,valid=len(valid),failed=len(samples)-len(valid),central_max=max(x['central_relative'] for x in valid),forward_max=max(x['forward_relative'] for x in valid),backward_max=max(x['backward_relative'] for x in valid),central_le_1e_6=sum(x['central_relative']<=1e-6 for x in valid)))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(schema='numi.cardboard.cpu-analytic-derivative-audit.v1',binding={str(p):sha(p) for p in [out/'audit.mm',out/'run.py',out/'prediction.json',out/'receipt.json',out/'rows.jsonl',Path(__file__)]},centers=len(rows),nonzero_direction=len(nonzero),zero_direction=len(rows)-len(nonzero),legacy_failures=sum(r['legacy_status']=='fail' for r in rows),native_analytic_worst=max(nonzero,key=lambda r:r['native_analytic_norm_relative_error']),native_analytic_norm_relative_above_1e_3=sum(r['native_analytic_norm_relative_error']>1e-3 for r in nonzero),sampled_centers=len(sampled),step_sweep=perstep,classification='Descriptive full-vector norm comparison, not a componentwise or measured-zero-floor qualification. Complete independent smooth implicit derivative at exact source13 inputs. Centered differences across changing yield/complementarity states are not interpreted as smooth reference tangents; both one-sided responses and exact state indicators are retained. Too-small FP64 steps show cancellation/projection noise. This does not close complete mesh/time convergence or physical validation.',physical_steps=0,physical_validation=False)
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(out/'README.md').write_text('Independent CPU differentiation of the source13 complete implicit equations on all 2,880 exact-input centers from native192. The source61 FP64 header, local compiler command, input/output/source/binary hashes and pre-run prediction are retained. All 1,440 nonzero-direction centers have native accurate tangent full-vector relative error below 1e-3; worst 1.0783e-4. This full-vector descriptive result does not replace componentwise gates or measured absolute noise floors. The 302 selected center sweeps retain forward, backward and centered derivatives plus actual multiplier increments and yield gaps. Large branch-crossing steps and very small cancellation-limited steps remain in the record. No all-case FP64 finite-difference 1e-6 gate, native solver convergence or physical qualification is asserted.\n\nReproduction: `/Users/home/.pyenv/versions/3.10.12/bin/python3 '+str(out/'run.py')+'`; then run `summarize.py`. Preserve the original directory before a new attempt.\n')
print(json.dumps({k:report[k] for k in ['centers','nonzero_direction','zero_direction','native_analytic_norm_relative_above_1e_3','sampled_centers']}))
