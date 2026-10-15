from pathlib import Path
import json,hashlib
B=Path('/Users/home/cardboard-implementation-20261014');N=B/'native/c1-trace-capacity32-shader-validation-long-403';O=B/'analysis/c1-native403-trace-capacity-verification-001';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def strict(s):return json.loads(s,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
r=strict((N/'receipt.json').read_text());assert r['exit_code']==0 and not r['timed_out'] and r['source_inputs_unchanged']
rows=[strict(line) for line in (N/'stdout.log').read_text().splitlines() if line.startswith('{')];traces=[x for x in rows if x.get('schema')=='numi.cardboard.c1-local-projection-trace.v1'];assert len(traces)==2
for t in traces:
 assert t['trace_supported'] and t['converged'] and t['native_projection_valid'] and t['native_stress_tangent_valid']
 assert len(t['iterations'])==t['attempted_iterations'] and t['attempted_iterations']<=32
 assert [i['index'] for i in t['iterations']]==list(range(t['attempted_iterations']))
assert sorted(t['attempted_iterations'] for t in traces)==[7,20]
assert 'Metal API Validation Enabled' in (N/'stderr.log').read_text();assert 'Metal GPU Validation Enabled' in (N/'stderr.log').read_text()
report=dict(status='PASS_NATIVE_COMPLETE_C1_TRACE_API_AND_SHADER_VALIDATION',receipt=r,stdout_sha256=sha(N/'stdout.log'),stderr_sha256=sha(N/'stderr.log'),host_trace_capacity=32,metal_trace_capacity=32,trace_float_words=11976,material_results=[dict(material=t['material'],attempted_iterations=t['attempted_iterations'],trace_records=len(t['iterations']),residual=t['final_residual_inf_scaled']) for t in traces],shader_validation='native400timeout retained; separate403 passes API and GPU shader validation',physical_steps=0,physical_validation=False)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'verify.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n');print(report['status'])
