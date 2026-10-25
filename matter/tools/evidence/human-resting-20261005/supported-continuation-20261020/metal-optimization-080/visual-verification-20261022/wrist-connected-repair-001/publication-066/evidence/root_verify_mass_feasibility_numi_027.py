from pathlib import Path
import json,math,hashlib,platform
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");P=E/'o12-outward-repair-027-f32-gated/mass-feasibility-001'
frozen=json.loads((P/'audit-report027-frozen.json').read_text());actual=json.loads((P/'root-ssh-numi-report.json').read_text());differences=[]
def compare(a,b,path):
 if isinstance(a,dict):
  assert set(a)==set(b),path
  for k in a:compare(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y)in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
 elif isinstance(a,float):
  assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(path,a,b)
  if a!=b:differences.append({'path':path,'ssh_numi':a,'prior':b,'absolute_difference':abs(a-b)})
 else:assert a==b,(path,a,b)
for key in ['totals','material_geometry','row_mapping','donor_capacity','configured_limits']:compare(actual[key],frozen[key],key)
receipt={'status':'SSH_NUMI_RERUN_NUMERICAL_PARITY_PASS_IDENTICAL_DECISIONS','hostname':platform.node(),'device':platform.platform(),'input_count':len(frozen['pinned_inputs']),'physical_qualification':False,'accepted_physical_steps':0,'report_sha256':hashlib.sha256((P/'root-ssh-numi-report.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((P/'audit_mass_027.py').read_bytes()).hexdigest(),'J_violations':actual['totals']['J_violations'],'comparison_tolerance':{'absolute':1e-12,'relative':1e-12},'nonbitwise_scalar_differences':differences,'initial_exact_json_comparison':'Failed only on last-bit floating sums, retained source and diagnostics; classified decisions and counts unchanged.'}
(P/'root-ssh-numi-parity.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

