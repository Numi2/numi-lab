from pathlib import Path
import json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
d=q/'checkout-publication032/matter/examples/cardboard/evidence/native-implementation-20261014'
old={r['case'] for r in json.loads((d/'summary.json').read_text())['records']}
exclude_dirs={'__pycache__','archives','base','build','restore-check','restore-test','inputs','CMakeFiles'}
cases=[];excluded=[]
for folder in ['analysis','native','solver']:
 for p in sorted((q/folder).iterdir()):
  if not p.is_dir():continue
  n=str(p.relative_to(q))
  if n in old:continue
  reason=None
  if n in ['analysis/native521-fresh-damped-coupon-command-002','solver/fresh-damped-cgls-complete-coupon-521']:reason='active native521 trajectory or mutable command package'
  elif p.name.startswith('source199'):reason='ongoing source199 checker repair'
  elif p.name.startswith('publication033') and not p.name.startswith('publication033-milestone-draft'):reason='ongoing publication033 preparation or independent audit'
  files=[f for f in p.rglob('*') if f.is_file() and not any(s in exclude_dirs or s.startswith('.air') for s in f.relative_to(p).parts[:-1])]
  if not reason and not files:reason='empty placeholder or only duplicate excluded trees'
  if reason:excluded.append({'case':n,'reason':reason});continue
  cases.append(n)
o=q/'analysis/publication033-case-inventory-001';assert not o.exists();o.mkdir();cases.append(str(o.relative_to(q)));cases.sort()
selection={'schema':'numi.cardboard.reference033-case-selection.v1','cases':cases,'excluded_candidates':excluded,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'active521_excluded':True,'source199_ongoing_excluded':True}
b=(json.dumps(selection,indent=2)+'\n').encode()
(q/'publication033-case-selection.json').write_bytes(b);(o/'selection.json').write_bytes(b);(o/'select.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'cases':len(cases),'excluded':excluded,'selection_sha256':hashlib.sha256(b).hexdigest()},indent=2))
