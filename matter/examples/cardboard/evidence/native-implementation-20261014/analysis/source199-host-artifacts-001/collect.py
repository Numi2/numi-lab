
from pathlib import Path
import json,hashlib,shutil,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
p=q/'analysis/source199-host-artifacts-001'
assert not p.exists();p.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
records=[]
def copy(src,name):
 src=Path(src);data=src.read_bytes();dest=p/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert dest.read_bytes()==data
 else:dest.write_bytes(data)
 records.append({'source_path':str(src),'bundle_path':name,'bytes':len(data),'sha256':sha(data)})
for name in ['source-contact-damped-memory-199-001','source-contact-damped-memory-199-buildfix-002','source-contact-damped-memory-199-buildfix-003']:
 root=q/'snapshots'/name
 if not root.exists():continue
 for f in ['frozen-source.json','matter/tools/cardboard_full_contact_armijo_check.mm']:
  copy(root/f,'snapshots/'+name+'/'+f)
bpath=q/'analysis/source199-postcommit-fault-signature-002/source-build.json'
b=json.loads(bpath.read_text());copy(bpath,'source-build.json')
for key,r in b['artifacts'].items():
 fp=Path(r['path']);assert fp.stat().st_size==r['size_bytes'] and sha(fp.read_bytes())==r['sha256']
 copy(fp,'artifacts/'+key+'/'+fp.name)
manifest={'schema':'numi.cardboard.source199-host-artifacts.v1','records':records,'canonical_frozen_sha256':sha((q/'snapshots/source-contact-damped-memory-199-buildfix-002/frozen-source.json').read_bytes()),'canonical_build_sha256':sha(bpath.read_bytes()),'qualification':'Source-bound strict regression checker only. Failed snapshot001 and rejected map003 retained; only canonical snapshot002 admitted.','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('files',len(records),'bytes',sum(r['bytes'] for r in records),'manifest',sha((p/'manifest.json').read_bytes()))
