#!/usr/bin/env python3
import hashlib,json,sys,subprocess
from pathlib import Path
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
out=Q/'analysis/source260-605-610-strict-replay-and-json-correction-001'
r=json.loads((out/'report.json').read_text())
assert r['gpu_dispatched'] is False and len(r['cases'])==2
for c in r['cases']:
 assert c['native']['hit'] is False and c['native']['certified'] is False
 assert c['reference']['exact_binary_rational_endpoint_sign_test']['strict_opposite_signs'] is True
old=Path(r['review003_json_correction']['original_path']).read_bytes()
fix=Path(r['review003_json_correction']['corrected_path']).read_bytes()
assert hashlib.sha256(old).hexdigest()==r['review003_json_correction']['original_sha256']
assert hashlib.sha256(fix).hexdigest()==r['review003_json_correction']['corrected_sha256']
assert json.loads(old[:-2])==json.loads(fix)
dec=Q/'analysis/source257-mortar-query-decoder-001'
p=subprocess.run([str(Q/'venvs/cardboard-reference-185/bin/python3'),str(dec/'decode_mortar_query.py'),'--self-test'],capture_output=True,text=True,check=True)
assert json.loads(p.stdout)['passed']==11
print(json.dumps({'strict_json':True,'cases':2,'strict_endpoint_crossings':2,'native_ccd_certified':0,'decoder_self_tests':11,'correction_semantics':'unchanged'},sort_keys=True))
