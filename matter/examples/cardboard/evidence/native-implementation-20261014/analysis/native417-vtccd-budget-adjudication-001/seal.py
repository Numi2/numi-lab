from pathlib import Path
import json,hashlib
B=Path('/Users/home/cardboard-implementation-20261014');I=B/'native/exact-vtccd-budget-replay-417';O=B/'analysis/native417-vtccd-budget-adjudication-001';O.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=json.loads((I/'receipt.json').read_text());j=json.loads((I/'stdout.log').read_text());assert r['exit_code']==0 and not r['timed_out'] and r['source_inputs_unchanged'];out=[]
for c in j['captures']:
 bs=c['budgets'];assert [x['budget'] for x in bs]==[16,32,64,128];assert bs[0]['return']==c['captured_return']
 for x in bs:assert x['iterations']==len(x['trace'])
 assert all(x['return']==bs[1]['return'] for x in bs[1:])
 out.append(dict(capture=c['capture'],manifest_sha256=c['capture_manifest_sha256'],payload_sha256=c['payload_sha256'],budget16_return_bitwise_match=True,budgets=[{k:x[k] for k in ['budget','return','iterations','certified','hit','exhausted_uncertified']} for x in bs]))
(O/'report.json').write_text(json.dumps(dict(schema='numi.cardboard.vtccd-budget-adjudication.v1',receipt_sha256=sha(I/'receipt.json'),native_output_sha256=sha(I/'stdout.log'),production_default_budget_unchanged=16,physical_transactions=0,results=out,verdict='Exact retained helper replay only. Trial21 changes from uncertified exhaustion at16 to certified hit at21. Trial22 is already a certified hit at15. Both are contact-rejecting outcomes; a larger budget does not establish full nonlinear fixture convergence.',qualification=False),indent=2)+'\n');(O/'seal.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n')
