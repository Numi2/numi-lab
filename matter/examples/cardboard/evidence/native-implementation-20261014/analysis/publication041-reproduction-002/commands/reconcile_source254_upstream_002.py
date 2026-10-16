from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');old=q/'snapshots/source-integrated-measured-coupon-contact-254-001';s=q/'snapshots/source-integrated-measured-coupon-contact-254-002';r=q/'checkout-publication032';d=q/'analysis/source254-upstream-reconciliation-002';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not s.exists() and not d.exists();s.mkdir();d.mkdir();f=json.loads((old/'frozen-source.json').read_text());cfg=json.loads((old/'configure-support-source.json').read_text())
upstream=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip();assert upstream=='b18cf2bb62ed492ab307ece760b4ec108fd37f69';assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=r)
n='apps/NumiHumanRestingVisual.hpp'
for name,h in {**cfg['files'],**f['source_sha256']}.items():
 p=old/name;assert sha(p)==h,name;z=s/name;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,z)
assert n in f['source_sha256'] and n in cfg['files'];before=sha(s/n);shutil.copy2(r/n,s/n);after=sha(s/n);assert before!=after
f['source_sha256'][n]=after;cfg['files'][n]=after
f['snapshot_label']=s.name;f['source254_upstream_reconciliation']={'base_snapshot':str(old),'base_frozen_sha256':sha(old/'frozen-source.json'),'upstream_revision':upstream,'changes':{n:{'before':before,'after':after}},'all_cardboard_source_bytes_unchanged':True,'purpose':'Retain concurrent Human accepted cardiac input capture update; no unrelated upstream work is overwritten.'}
for name,obj in [('frozen-source.json',f),('configure-support-source.json',cfg)]: (s/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
assert all(sha(s/n)==h for n,h in f['source_sha256'].items())
rec={'schema':'numi.cardboard.source254-upstream-reconciliation.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'configure_support_sha256':sha(s/'configure-support-source.json'),'upstream_revision':upstream,'updated_path':n,'before_sha256':before,'after_sha256':after,'unchanged_selected_files':len(f['source_sha256'])-1,'first_composition_build_all_pass':json.loads((q/'analysis/source254-build-001/report.json').read_text())['all_pass'],'new_build_pending':True,'GPU_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
