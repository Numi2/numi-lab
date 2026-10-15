from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-bounded-native-cgls-mask-repair-142';O=B/'analysis/source142-root-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((B/'analysis/native450-cgls-adjudication-001/report.json').read_text())
p=json.loads((B/'analysis/native450-capture-off-parity-001/report.json').read_text())
assert p['parity_pass'] and a['native450_complete_attempt_failed']
base=json.loads((B/'snapshots/source-coupon-cache-admission-139/frozen-source.json').read_text())['source_sha256']
current=json.loads((S/'frozen-source.json').read_text())['source_sha256']
changes=['matter/CMakeLists.txt','matter/src/metal/base_cgls.metalinc','matter/src/metal/base_cgls_runtime.metalinc','matter/src/metal/fgmres.metalinc','matter/src/runtime.mm']
assert sorted(k for k in set(base)|set(current) if base.get(k)!=current.get(k))==sorted(changes)
for k,v in base.items(): assert sha(R/k)==v, k
for k,v in current.items(): assert sha(S/k)==v,k
O.mkdir(exist_ok=False)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
for k in changes: shutil.copy2(S/k,R/k)
for k,v in current.items(): assert sha(R/k)==v,k
record={'schema':'numi.cardboard.source142_root_integration.v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_root_head':head,'source_snapshot':str(S),'frozen_source_sha256':sha(S/'frozen-source.json'),'all_787_selected_source_hashes_match':True,'changed_files':{k:sha(R/k) for k in changes},'native450_root_adjudication_sha256':sha(B/'analysis/native450-cgls-adjudication-001/report.json'),'native450_452_observer_parity_sha256':sha(B/'analysis/native450-capture-off-parity-001/report.json'),'default_off':True,'complete_coupon_qualified':False,'physical_validation':False,'qualification_scope':'Synthetic helper678checks; selected actualFEM A/A-transpose actions; exactN48nonlineartrial; fullobserver attempted-historyparity. Source141 feature-off fullhistory comparison is retained separately, not claimed as a freshsource142disabled run.'}
(O/'report.json').write_text(json.dumps(record,indent=2)+'\n');(O/'integrate.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.integration_seal.v1','files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps(record,indent=2))
