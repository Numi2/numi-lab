from pathlib import Path
import hashlib,json,shutil,datetime
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
old=B/'solver/bounded-cgls64-exact-state-450';new=B/'solver/cgls-primary-feature-off-parity-454';O=B/'analysis/primary-feature-off454-inputs-001';O.mkdir(exist_ok=False)
c=json.loads((B/'native-450-command.json').read_text())
c=[x.replace(str(old),str(new)).replace('source-bounded-native-cgls-mask-repair-142','source-cgls-primary-policy-143').replace('build-cardboard-provenance-142','build-cardboard-provenance-143') for x in c]
c[c.index('--question')+1]='Does source143 with the primary policy absent preserve the complete attempted source142450 history?'
c[c.index('--prediction')+1]='Expect identical compiled physical world and every accepted/rejected physical history, ordinary Armijo and true-residual records, CGLS action fields and rollback through terminal receipt after excluding only source/runtime provenance and metadata. The source143 primary policy is absent; CGLS64 remains fallback at exactly control1/Newton48. Any mismatch blocks default-off parity. Both full coupon attempts may fail; parity cannot qualify coupon completion or convergence.'
files=[(old/'receipt.json','source142450-baseline-receipt.json'),(B/'analysis/source143-primary-policy-independent-review-001/report.json','source143-independent-review.json'),(B/'analysis/source143-host-policy-parser-check-001/report.json','source143-parser-report.json')]
insert=[];bindings=[]
for src,name in files:
 dst=O/name;shutil.copy2(src,dst);assert dst.read_bytes()==src.read_bytes();insert+=['--input-file',str(dst)];bindings.append({'source':str(src),'alias':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
(O/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n');insert+=['--input-file',str(O/'bindings.json')];c[c.index('--'):c.index('--')]=insert
assert not new.exists();assert all(not x.startswith('NM_FEM_BASE_CGLS_PRIMARY_AFTER=') for x in c)
(B/'native-454-command.json').write_text(json.dumps(c,indent=2)+'\n');(O/'prepare.py').write_bytes(Path(__file__).read_bytes())
(O/'protocol.json').write_text(json.dumps({'schema':'numi.cardboard.primary_feature_off454_protocol.v1','stage':'PREPARED_NOT_EXECUTED','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command_sha256':hashlib.sha256((B/'native-454-command.json').read_bytes()).hexdigest(),'source_snapshot':str(B/'snapshots/source-cgls-primary-policy-143'),'all_physical_and_solver_parameters_unchanged_from450':True,'physical_validation':False},indent=2)+'\n')
guard=(B/'run_native450.py').read_text().replace('native-450-command.json','native-454-command.json').replace('native450-launch-guard-001','native454-launch-guard-001').replace('native450-runner.log','native454-runner.log');(B/'run_native454.py').write_text(guard)
print('454 prepared',hashlib.sha256((B/'native-454-command.json').read_bytes()).hexdigest())
