from pathlib import Path
import hashlib,json
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
O=B/'analysis/closed-native479480483-transfer-inventory-001'
assert not O.exists();O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cases={
 '479':('solver/warm-cgls-control19-checkpoint-479','analysis/native479-control19-checkpoint-command-001/execution-479/receipt.json'),
 '480':('solver/warm-cgls-control19-checkpoint-480','analysis/native480-control19-checkpoint-command-001/execution-480/receipt.json'),
 '483':('solver/rejected-encoder-cleanup-483','analysis/native483-rejected-encoder-cleanup-command-001/execution-483/receipt.json'),
}
result={}
for k,(root,receipt) in cases.items():
 p=B/receipt;r=json.loads(p.read_text());assert r['execution_started'] and not r['errors'] and not r['timed_out'] and not r['space_abort']
 files=[{'path':str(f.relative_to(B/root)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted((B/root).rglob('*')) if f.is_file()]
 result[k]={'root':str(B/root),'terminal_receipt_path':str(p),'terminal_receipt_sha256':sha(p),'files':files}
(O/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
(O/'inventory.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'inventory_sha256':sha(O/'inventory.json'),'files':{k:len(v['files']) for k,v in result.items()}}))
