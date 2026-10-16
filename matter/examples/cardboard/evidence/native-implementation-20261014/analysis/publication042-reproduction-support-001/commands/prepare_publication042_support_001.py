from pathlib import Path
import json,hashlib,shutil,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-integrated-globalization-contact-memory-259-003';r=q/'checkout-publication032';d=q/'analysis/publication042-reproduction-support-001';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();items={}
def put(p,n):
 z=d/n;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,z);assert sha(z)==sha(p);items[str(n)]={'sha256':sha(p),'bytes':p.stat().st_size,'external_path':str(p)}
support=json.loads((s/'configure-support-source.json').read_text())['files'];exceptions=[]
for n,h in support.items():
 if not (r/n).is_file() or sha(r/n)!=h:assert sha(s/n)==h;put(s/n,Path('canonical-support')/n);exceptions.append(n)
for p in sorted(q.glob('*publication042*.py')):put(p,Path('commands')/p.name)
report={'schema':'numi.cardboard.publication042-reproduction-support.v1','canonical_frozen_sha256':sha(s/'frozen-source.json'),'support_exceptions':exceptions,'scope':'Canonical support bytes absent from the checkout (historical archive-only worker lock) and exact publication/recovery commands. Zero-byte archived locks confer no current worker ownership.','files':items,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}},indent=2)+'\n');print(json.dumps({'support_exceptions':exceptions,'files':len(items),'report_sha256':sha(d/'report.json')}))
