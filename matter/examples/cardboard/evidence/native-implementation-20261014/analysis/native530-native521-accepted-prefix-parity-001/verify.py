
from pathlib import Path
import json,hashlib,csv,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
a=q/'solver/accepted-prefix-checkpoint-530/output';b=q/'solver/fresh-damped-cgls-complete-coupon-521/output'
o=q/'analysis/native530-native521-accepted-prefix-parity-001';assert not o.exists();o.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=[p.name for p in a.iterdir() if p.is_file() and (p.suffix=='.obj' or p.name.startswith('material_state_'))]
rows=[]
for n in sorted(names):
 assert (b/n).is_file(),n
 equal=(a/n).read_bytes()==(b/n).read_bytes()
 rows.append({'name':n,'sha256_530':sha(a/n),'sha256_521':sha(b/n),'bytes':(a/n).stat().st_size,'byte_identical':equal})
assert all(r['byte_identical'] for r in rows)
la=(a/'observations.csv').read_bytes().splitlines(keepends=True);lb=(b/'observations.csv').read_bytes().splitlines(keepends=True)
assert len(la)==35 and la==lb[:35]
pr=json.loads((a/'coupon-checkpoint-prefix.json').read_text());assert pr['accepted_control_index']==16 and pr['next_control_index']==17 and pr['physical_validation'] is False
report={'schema':'numi.cardboard.native530-native521-prefix-parity.v1','verdict':'PASS','accepted_controls':17,'accepted_observation_rows_byte_identical':34,'accepted_geometry_and_material_files':rows,'checkpoint_sha256':sha(a/'checkpoints/coupon-control-000016.nmcb'),'metadata_sha256':sha(a/'checkpoints/coupon-control-000016.resume'),'prefix_sha256':sha(a/'checkpoints/coupon-control-000016.prefix'),'full_trajectory_qualified':False,'physical_qualification':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(o/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS files',len(rows),'report',sha(o/'report.json'))
