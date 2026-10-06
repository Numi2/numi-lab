"""Rebind an unchanged anatomical cardiac map after the isolated bladder and taenia repairs."""
from pathlib import Path
import hashlib,json,subprocess,sys

E=Path('/Users/n/numi-human-resting-evidence-20261005')
out=E/'bladder-cardiac-input-113';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-001/src')
from numilab_human.resting_pleura_proxy import _parse_payload,_record_content_bytes
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
old=E/'passive-organ-native-input-073';new=E/'bladder-native-input-112'
_,_,oldrows=_parse_payload((old/'resting-thorax.nhanatomy').read_bytes())
_,_,newrows=_parse_payload((new/'resting-thorax.nhanatomy').read_bytes())
ids=[1,23,24,318,319,320,321]
proof={}
for sid in ids:
    before=_record_content_bytes(oldrows[sid]);after=_record_content_bytes(newrows[sid])
    assert before==after,('cardiac geometry changed',sid)
    proof[str(sid)]=hashlib.sha256(after).hexdigest()
original=E/'cardiac-map-multiregion-affine-repair-011/ventricular-wall-map-refinement-fullchain073-002.json'
descriptor=json.loads(original.read_text())
descriptor['source_payload_sha256']=sha(new/'resting-thorax.nhanatomy')
descriptor['source_receipt_sha256']=sha(new/'resting-anatomy-receipt.json')
path=out/'ventricular-wall-map-refinement.json';path.write_text(json.dumps(descriptor,separators=(',',':'))+'\n')
helper=Path('/Users/n/numi-human-resting-cardiac-source-026/matter/tools/cardiac_geometry_binding.py')
argv=[sys.executable,str(helper),'--input',str(new/'resting-thorax.nhanatomy'),
      '--input-receipt',str(new/'resting-anatomy-receipt.json'),
      '--arrangement-candidate',str(E/'cardiac-wall-resolution-003/candidate.json'),
      '--wall-map-refinement',str(path),'--output',str(out/'resting-thorax.nhanatomy'),
      '--output-receipt',str(out/'resting-anatomy-receipt.json'),'--attach-map-refinement']
inv={'argv':argv,'driver_sha256':sha(__file__),'helper_sha256':sha(helper),
     'original_descriptor':str(original),'original_descriptor_sha256':sha(original),
     'descriptor_changed_fields':['source_payload_sha256','source_receipt_sha256'],
     'byte_identical_cardiac_record_sha256':proof,'input_sha256':sha(new/'resting-thorax.nhanatomy'),
     'input_receipt_sha256':sha(new/'resting-anatomy-receipt.json')}
with (out/'attachment.log').open('w') as log:
    result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
inv['return_code']=result.returncode
if result.returncode==0:
    inv['output_sha256']=sha(out/'resting-thorax.nhanatomy')
    inv['output_receipt_sha256']=sha(out/'resting-anatomy-receipt.json')
    assert inv['output_sha256']==inv['input_sha256']
(out/'invocation.json').write_text(json.dumps(inv,indent=2)+'\n')
print(json.dumps(inv),flush=True)
raise SystemExit(result.returncode)
