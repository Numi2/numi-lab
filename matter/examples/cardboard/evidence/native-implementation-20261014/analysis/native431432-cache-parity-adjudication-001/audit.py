from pathlib import Path
import json,hashlib
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');O=B/'analysis/native431432-cache-parity-adjudication-001';O.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binding=B/'analysis/cache-reuse-native431432-inputs-001/fixture-bindings.json';j=json.loads(binding.read_text())
for p,h in j['files'].items():assert sha(Path(p))==h,p
build=R/'build-cardboard-provenance-134/source-build.json';b=json.loads(build.read_text());assert b['success'];rows=[]
for fixture,n in ((360,431),(419,432)):
 case=B/f'native/tangent-cache-reuse-run{fixture}-{n}';receipt=json.loads((case/'receipt.json').read_text());result=case/'result/projection-reuse-parity.json';v=json.loads(result.read_text())
 assert receipt['execution_started'] and receipt['exit_code']==0 and not receipt['timed_out'] and receipt['source_inputs_unchanged']
 assert v['exact_cache_and_action_parity'] and v['physical_validation'] is False and v['performance_claim'] is False
 for k in ('validity_bit_differences','matrix_bit_differences','historical_matrix_bit_differences','captured_correction_action_bit_differences','random_action_bit_differences','matrix_nonfinite_values','captured_correction_action_nonfinite_values','random_action_nonfinite_values'):assert v[k]==0,k
 assert v['elements']==3840 and v['old_valid_entries']==3840 and v['reuse_valid_entries']==3840 and v['matrix_float_comparisons']==3840*81
 rows.append(dict(fixture=fixture,case=n,result_sha256=sha(result),receipt_sha256=sha(case/'receipt.json'),word_parity=True,physical_steps=0))
report=dict(schema='numi.cardboard.cache_projection_reuse_native_parity.v1',source134_build_sha256=sha(build),input_binding_sha256=sha(binding),cases=rows,interpretation='Diagnostic sibling reuses one accurate 15-state projection across nine local tangent columns. Both exact captured states preserve original production cache validity, all matrix words, captured correction and deterministic random element-action words.',limits=['No production adoption in main.','Rate-independent materials; nonzero deformation rates do not supply rate-dependent coverage.','No full-cycle history comparison or kernel speed measurement.','No solver, convergence, damage calibration, assembly or impact qualification.'],retained_failures=['native423 wrong expected parameter count','native427 malformed compiled-package hash literal','native428 eager nil NSError string conversion crash','source134 initial wrapper -p1 admission failure before mutation'],physical_transactions=0)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2)+'\n');print(json.dumps(report))
