from pathlib import Path
import json,hashlib,csv,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');a=q/'solver/fresh-source210-strict-inner-full-coupon-545/output';b=q/'solver/control19-newton3-resume-diagnostic-555/output';d=q/'analysis/native555-production-suffix-parity-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:list(csv.DictReader(p.open()))
x=read(a/'observations.csv');y=read(b/'observations.csv');suffix=[r for r in x if int(r['step'])>=16]
different=[{'step':r['step'],'environment':r['environment'],'columns':{k:[r[k],s[k]] for k in r if r[k]!=s[k]}} for r,s in zip(suffix,y) if r!=s]
assert len(suffix)==len(y)==8
assert not different,different
files=sorted(n.name for n in b.iterdir() if n.is_file() and (n.name.startswith('accepted_step_') or n.name.startswith('rejected_step_') or n.name.startswith('material_state_') or n.name in ['compiled.nmatterpack','mesh.json','manifest.json']))
records=[]
for n in files:
 assert (a/n).is_file(),n
 same=(a/n).read_bytes()==(b/n).read_bytes();records.append({'path':n,'original_sha256':sha(a/n),'replay_sha256':sha(b/n),'byte_identical':same});assert same,n
bins=[]
for p in sorted((b/'rejected-conditioning-000020').glob('*.bin')):
 n=p.name;aa=a/'rejected-conditioning-000020'/n
 same=aa.read_bytes()==p.read_bytes();bins.append({'path':n,'original_sha256':sha(aa),'replay_sha256':sha(p),'byte_identical':same});assert same,n
r={'schema':'numi.cardboard.native555-production-suffix-parity.v1','verdict':'PASS_EXACT_RESUMED_SUFFIX','original_case':'solver/fresh-source210-strict-inner-full-coupon-545','diagnostic_case':'solver/control19-newton3-resume-diagnostic-555','original_observations_sha256':sha(a/'observations.csv'),'replay_observations_sha256':sha(b/'observations.csv'),'compared_control_steps':[16,17,18,19],'rows':len(y),'all_csv_fields_identical':True,'files':records,'rejected_conditioning_bins':bins,'accepted_all_environment_controls':[16,17,18],'failure_reproduced_at_control':19,'capture_sha256':'c0a111c510ef4c5891ace3bb58f65af48740733d7b46987be370561d1c22ab53','limits':'Original accepted checkpoint15 supplies the prefix. Comparison covers complete resumed production observations, accepted/rejected geometry, final material histories, and saved rejected-conditioning binary state. It is diagnostic observer parity, not completed coupon qualification. Result/source/path metadata are not production-state comparisons.','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'verdict':r['verdict'],'rows':len(y),'files':len(records),'bins':len(bins),'report_sha256':sha(d/'report.json')}))
