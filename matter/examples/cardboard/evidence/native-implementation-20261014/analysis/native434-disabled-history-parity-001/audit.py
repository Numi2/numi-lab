from pathlib import Path
import hashlib, importlib.util, json
B=Path('/Users/home/cardboard-implementation-20261014'); O=B/'analysis/native434-disabled-history-parity-001'
A=B/'solver/two-stage-coupon-first-descent-capture-415'; C=B/'solver/base-cauchy-disabled-parity-434'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('history',B/'analysis/native419-history-parity-001/audit_419.py'); h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
for root in [A,C]:
 r=json.loads((root/'receipt.json').read_text());assert r['execution_started'] and r['source_inputs_unchanged'] and r['exit_code']==2 and not r['timed_out']
rows=h.compare_tree(A/'output',C/'output'); diffs=[x for x in rows if not x['equal']]
assert len(rows)==35
assert [d['path'] for d in diffs]==['manifest.json','rejected-conditioning-000002/manifest.json'],diffs
assert {d['field'] for d in diffs[0]['json_differences']}=={'$/materials/glue_path','$/materials/liner_path','$/materials/medium_path'}
assert [d['field'] for d in diffs[1]['json_differences']]==['$/device_program_fingerprint']
assert (A/'stdout.log').read_bytes()==(C/'stdout.log').read_bytes()
assert (A/'first-status5.bin').read_bytes()==(C/'first-status5.bin').read_bytes()
def events(root,prefix):return [x for x in (root/'stderr.log').read_text().splitlines() if x.startswith(prefix)]
def arm(root):
 out=[]
 for line in events(root,'fem_residual_armijo_iteration='):
  j=json.loads(line.split('=',1)[1]);kind=j.pop('direction_kind',None)
  assert kind in [None,'newton']
  # The only changed prose labels the scratch status more precisely. Preserve
  # and compare every numerical/status field, including those private values.
  j['semantics'].pop('trial_status');out.append(j)
 return out
assert len(arm(A))==len(arm(C))==64 and arm(A)==arm(C)
ta=events(A,'fem_true_residual_cycle=');tc=events(C,'fem_true_residual_cycle=');assert len(ta)==len(tc)==4096 and ta==tc
excluded=('cardboard_step_timing=','fem_coupled_capture_','fem_status5_capture_','fem_residual_armijo_iteration=')
clean=lambda root:[x for x in (root/'stderr.log').read_text().splitlines() if not x.startswith(excluded)]
assert clean(A)==clean(C)
_,pa,ea=h.command_parts(B/'native-415-command.json');_,pc,ec=h.command_parts(B/'native-434-command.json')
assert h.normalized_probe(pa)==h.normalized_probe(pc)
for key in ['NM_FEM_STATUS5_CAPTURE_PATH','NM_FEM_COUPLED_CAPTURE_PATH']:ea.pop(key);ec.pop(key)
assert ea==ec and 'NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR' not in ec
O.mkdir();report=dict(schema='numi.cardboard.disabled_cauchy_history_parity.v1',verdict='pass_for_exact_attempted_history; complete_coupon_remains_failed',output_files=len(rows),output_byte_exact=sum(r['equal'] for r in rows),metadata_differences=diffs,armijo_records=64,true_residual_records=4096,all_numerical_trace_fields_exact=True,all_remaining_stderr_exact=True,stdout_exact=True,first_status5_payload_exact=True,physical_probe_argv_equal_after_only_path_normalization=True,physical_environments_equal=True,runs={str(p):dict(receipt_sha256=sha(p/'receipt.json'),stderr_sha256=sha(p/'stderr.log'),result_sha256=sha(p/'output/result.json')) for p in [A,C]},claim_limit='The disabled fallback preserves source125/415 accepted/rejected history exactly. Both runs reject loading control1 after one accepted control; no complete coupon or convergence qualification.')
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps(dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in O.iterdir() if p.is_file()}),indent=2)+'\n');print('native434parity',report['output_byte_exact'],len(rows),flush=True)
