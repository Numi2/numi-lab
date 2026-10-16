
from pathlib import Path
import json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
p=q/'analysis/native528529-independent-strict-fault-audit-001'
assert not p.exists();p.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
rows=[]
for new,old in [('strict199-regional-528','integrated198-contact-regional-526'),('strict199-scalar-negative-529','integrated198-contact-restart16-527')]:
 a=q/'native'/new;b=q/'native'/old
 af={str(f.relative_to(a)):f.read_bytes() for f in a.rglob('*') if f.is_file()}
 bf={n:(b/n).read_bytes() for n in af}
 assert af==bf,(new,'physical outputs differ')
 for i in range(3):
  x=json.loads(af['baseline/transaction-%06d.json'%i])
  assert x['accepted']==(i<2)
  if i==2:
   status=x['after']['statuses'][0]
   assert status['code']==(1 if new.endswith('528') else 10)
   if new.endswith('528'):
    assert status['environment']==0 and status['object_index']==4294967295 and status['failing_index']==0 and status['diagnostics']==[-3,-3,-3,-3]
   else:assert status['failing_index']==18
   assert x['before']['accepted_state_vector_sha256']==x['after']['accepted_state_vector_sha256']
   assert x['before']['control_step']==x['after']['control_step']==2
   assert x['after']['solver_certificates'][0]['validity'][3]==0
 rows.append({'case':new,'comparison':old,'byte_identical_files':len(af),'bytes':sum(map(len,af.values())),'case_file_hashes':{n:sha(v) for n,v in af.items()},'rejected_status':status})
ep=q/'analysis/native528529-strict-postcommit-fault-command-001'
receipts={}
for n in ['528','529']:
 r=json.loads((ep/('execution-'+n)/'receipt.json').read_text())
 assert r['exit_code']==(0 if n=='528' else 1) and not r['errors'] and not r['timed_out'] and not r['space_abort'] and r['source_and_artifacts_unchanged']
 receipts[n]={'sha256':sha((ep/('execution-'+n)/'receipt.json').read_bytes()),'exit_code':r['exit_code']}
stderr=(ep/'execution-529/stderr.log').read_text()
assert 'postCommit fault signature mismatch' in stderr and 'got code=10 environment=0 object=0 failing_index=18' in stderr
report={'schema':'numi.cardboard.native528529-independent-strict-fault-audit.v1','verdict':'PASS','qualification':'Bounded synthetic regression and negative control only. No full coupon, convergence or box qualification.','cases':rows,'receipts':receipts,'negative_control_rejected_for_intended_reason':True,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(p/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(p/'verify.py').write_bytes(Path(__file__).read_bytes())
print(str(p),'report_sha256',sha((p/'report.json').read_bytes()))
