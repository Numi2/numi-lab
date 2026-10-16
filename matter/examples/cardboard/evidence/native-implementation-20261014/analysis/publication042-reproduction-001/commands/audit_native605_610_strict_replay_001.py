from pathlib import Path
import json,hashlib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native605-610-independent-replay-root-audit-001';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pkg=q/'analysis/source260-605-610-strict-replay-and-json-correction-001';decoder=q/'analysis/source257-mortar-query-decoder-001/decode_mortar_query.py'
assert sha(pkg/'report.json')=='0a715529bca36eb1b4dc4bbfa0b28b60cee10946d46e72cd18f221bc68ddc34d'
reference=json.loads((pkg/'report.json').read_text());a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(decoder),'--out',str(d/'recomputed.json')]
p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(d/'stdout.log').write_bytes(p.stdout);(d/'stderr.log').write_bytes(p.stderr);assert p.returncode==0
j=json.loads((d/'recomputed.json').read_text());rows=[];inputs=[Path(__file__),decoder,pkg/'report.json',pkg/'manifest.json']
for c,ref in zip(j['cases'],reference['cases']):
 assert c['run']==ref['run']
 g=c['independent_geometry'];n=c['decoded'];fp=c['source_like_fp32_sample_at_native_t'];ca=c['source_like_fp32_ca']
 assert g['exact_binary_rational_endpoint_sign_test']==ref['reference']['exact_binary_rational_endpoint_sign_test']
 assert c['record_sha256']==ref['record_sha256'] and c['capture_json_sha256']==ref['capture_json_sha256'] and c['payload_sha256']==ref['payload_sha256']
 assert not n['native_hit'] and not n['native_certified'] and fp['distance_bits']==n['native_separation_bits']
 assert g['binary64_bisection_t']==ref['reference']['binary64_bisection_t']
 rows.append({'run':c['run'],'record_sha256':c['record_sha256'],'exact_binary_rational_endpoint_crossing':True,'fp32_cpu_source_order_sample_at_native_time_separation_bits_match':True,'full_fp32_recurrence_toi_equals_native':ca['outcome']['toi']==n['native_toi'],'recurrence_claim':'CPU emulation omits SAT branches; no complete Metal recurrence certificate is inferred, and610 trajectory parity remains inconclusive.','production_hit':False,'production_certified':False})
 inputs.extend([Path(c['capture_json_path']),Path(c['payload_path'])])
old=Path(reference['review003_json_correction']['original_path']);fix=Path(reference['review003_json_correction']['corrected_path']);assert json.loads(old.read_bytes()[:-2])==json.loads(fix.read_bytes());inputs.extend([old,fix])
report={'schema':'numi.cardboard.native605-610-independent-replay-root-audit.v1','verdict':'PASS_RECOMPUTED_EXACT_ENDPOINT_CROSSINGS_AND_NATIVE_TIME_SAMPLE_BITS','argv':a,'cases':rows,'corrected_json_semantically_equal':True,'GPU_execution':False,'production_CCD_or_whole_candidate_certificate':False,'scope':'Root reruns actual-capture decoder and exact rational endpoint distance calculation, compares complete rational records and captured hashes against the sealed package. These path crossings do not certify earliest TOI or full candidate admissibility.','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'inputs':{str(x.relative_to(q)):sha(x) for x in inputs},'artifacts':{x.name:sha(x) for x in d.iterdir() if x.is_file()}},indent=2)+'\n');print(json.dumps(report))
