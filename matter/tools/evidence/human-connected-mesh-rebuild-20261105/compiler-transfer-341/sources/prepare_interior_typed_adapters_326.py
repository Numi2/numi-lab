from pathlib import Path
import hashlib,json
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=Path(__import__('sys').argv[1]).resolve() if len(__import__('sys').argv)>1 else W/'interior-preserved-boundary-326';C=D/'compiler'
old=W/'joint-boundary-bone-admitted-320/compiler/typed-adapter';out=C/'typed-adapter';out.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
receipt={}
for name in ['typed-current-schema.py','active-current-volume.py','audit-role-orientation.py']:
 source=old/name;s=source.read_text()
 s=s.replace(str(old),str(out))
 if name=='active-current-volume.py':
  import re
  s,count=re.subn(r"ADAPTER_1021_SHA256 = [^\n]+", "ADAPTER_1021_SHA256 = "+repr(sha(out/'typed-current-schema.py')), s)
  assert count==1
 (out/name).write_text(s)
 receipt[name]={'source':str(source),'source_sha256':sha(source),'output_sha256':sha(out/name)}
(out/'rebind-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
