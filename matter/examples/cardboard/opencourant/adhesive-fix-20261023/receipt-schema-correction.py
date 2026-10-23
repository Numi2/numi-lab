from pathlib import Path
import hashlib,json,datetime
r=Path('/Users/n/cardboard-opencourant-20261023/adhesive-fix/post')
for p in r.glob('*/T*/conversion.json'):
 d=json.loads(p.read_text())
 if d.get('receipt_created_after_conversion'): continue
 original=p.with_name('conversion-initial.json');original.write_bytes(p.read_bytes())
 d['original_receipt_sha256']=hashlib.sha256(original.read_bytes()).hexdigest()
 d['receipt_created_after_conversion']=True
 d['receipt_created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 d['receipt_schema_correction']='Original producer wrote this receipt after subprocess conversion but omitted its required schema flag. Original receipt preserved.'
 p.write_text(json.dumps(d,indent=2)+'\n')
