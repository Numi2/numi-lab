from pathlib import Path
import json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
d=q/'checkout-publication032/matter/examples/cardboard/evidence/native-implementation-20261014'
oldjson=json.loads((d/'summary.json').read_text());old={r['case'] for r in oldjson['records']}
prior={f.get('repository_copy',f.get('repository_compressed_copy')):f for r in oldjson['records'] for f in r['files']}
exclude_dirs={'__pycache__','archives','base','build','restore-check','restore-test','inputs','CMakeFiles'}
cases=[];excluded=[];overlap=[]
for folder in ['analysis','native','solver']:
 for p in sorted((q/folder).iterdir()):
  if not p.is_dir():continue
  n=str(p.relative_to(q))
  if n in old:continue
  reason=None
  if n in ['analysis/native521-fresh-damped-coupon-command-002','solver/fresh-damped-cgls-complete-coupon-521']:reason='active native521 trajectory or mutable command package'
  elif p.name.startswith('source199'):reason='ongoing source199 checker repair'
  elif p.name.startswith('publication033') and not (p.name.startswith('publication033-milestone-draft') or p.name in ['publication033-payload-preflight-001','publication033-case-inventory-001']):reason='ongoing publication033 preparation or independent audit'
  files=[f for f in p.rglob('*') if f.is_file() and not any(s in exclude_dirs or s.startswith('.air') for s in f.relative_to(p).parts[:-1])]
  newfiles=[]
  if not reason:
   for f in files:
    target=Path(n)/f.relative_to(p)
    if f.stat().st_size>262144 and f.suffix!='.gz':target=target.with_name(target.name+'.gz')
    key=str(target)
    if key in prior:
     h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==prior[key]['sha256'] and f.stat().st_size==prior[key]['bytes'],key
     overlap.append({'case':n,'file':str(f.relative_to(p)),'historical_payload':key,'sha256':h})
    else:newfiles.append(f)
   if not newfiles:reason='empty placeholder, only duplicate excluded trees, or all files already indexed'
  if reason:excluded.append({'case':n,'reason':reason});continue
  cases.append(n)
o=q/'analysis/publication033-case-inventory-002';assert not o.exists();o.mkdir();cases.append(str(o.relative_to(q)));cases.sort()
selection={'schema':'numi.cardboard.reference033-case-selection.v2','cases':cases,'excluded_candidates':excluded,'verified_historical_overlaps':overlap,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'active521_excluded':True,'source199_ongoing_excluded':True}
b=(json.dumps(selection,indent=2)+'\n').encode()
(q/'publication033-case-selection.json').write_bytes(b);(o/'selection.json').write_bytes(b);(o/'select.py').write_bytes(Path(__file__).read_bytes())
p=q/'archive_reference033_v2.py';s=p.read_text().replace("raw+=1","israw=True").replace("compressed+=1","israw=False")
s=s.replace("put(D/target,payload);paths.append", "raw+=int(israw);compressed+=int(not israw)\n   put(D/target,payload);paths.append")
p.write_text(s)
print(json.dumps({'cases':len(cases),'overlapping_old_files':len(overlap),'excluded':excluded,'selection_sha256':hashlib.sha256(b).hexdigest()},indent=2))
