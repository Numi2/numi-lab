
from pathlib import Path
import hashlib,importlib.util,json,sys,time,traceback
import numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');C=W/'joint-boundary-rebuild-64c/compiler';B=W/'prescribed-local-mass-315';O=B/'admission-002';O.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
runpath=B/'allocation/run-manifest.json';run=json.loads(runpath.read_text());env=dict(a.split('=',1) for a in run['argv'] if a.startswith('NUMI_H706_FRACTIONAL_'))
cfg=json.loads((C/'mass-admission/identity.json').read_text())
cfg['h706_fractional_env']={k:env['NUMI_H706_FRACTIONAL_'+k] for k in cfg['h706_fractional_env']}
cfg['h706_fractional_environment_source']={**pin(runpath),'resolved_path':str(runpath.resolve()),'argv_sha256':hashlib.sha256(json.dumps(run['argv'],sort_keys=True,separators=(',',':')).encode()).hexdigest()}
cfg['h706_fractional_run_control_pins']={k:{field:run[k][field] for field in ('path','sha256')} for k in ['run_config','source_control_manifest']}
for p in [B/'source-control.json',B/'allocation/run-config.json',runpath,Path(run['prescribed_child_mass']['path'])]:cfg['input_pins'][str(p)]=sha(p)
cfg['output_root']=str(O/'unused-native-output');cfg['candidate_label']='64c geometry with prescribed unchanged-cell mass'
save(O/'identity.json',cfg)
module_path=C/'mass-admission/build_current_fractional_mass_receipt_1061.py'
spec=importlib.util.spec_from_file_location('retained_mass_admission_017',module_path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
start=time.time()
try:
 mass,projection=module.build_receipt_017(runpath,B/'allocation/execution.json',B/'allocation/stdout.log',O/'identity.json',B/'source-control.json',O/'mass-receipt.json')
 (O/'mass-receipt.json').write_bytes(module.canonical_receipt_bytes(mass))
 if projection:save(O/'source-projection.json',projection)
 # Independent target parity from actual recorded flow rows. General absolute
 # closure tolerances must not hide a large relative change in tiny cells.
 flow=np.loadtxt(B/'allocation/allocator-output/flow.csv',delimiter=',',skiprows=1,usecols=(0,2))
 actual=np.bincount(flow[:,0].astype(np.int64),weights=flow[:,1]);del flow
 target=np.loadtxt(run['prescribed_child_mass']['path'],delimiter=',',skiprows=1)
 rows=target[:,0].astype(np.int64);values=target[:,1]
 assert np.array_equal(rows.astype(float),target[:,0]) and np.all(np.diff(rows)>0)
 rel=np.abs(actual[rows]-values)/values;maximum=float(rel.max())
 assert maximum<=1e-10+64*np.finfo(float).eps,(maximum,int(rows[rel.argmax()]))
 save(O/'prescribed-parity.json',{'status':'PASS_INDEPENDENT_RECORDED_FLOW_TARGET_PARITY','prescribed_cells':len(rows),'maximum_relative_mass_difference':maximum,'maximum_absolute_mass_difference_kg':float(np.abs(actual[rows]-values).max()),'target':run['prescribed_child_mass'],'flow':pin(B/'allocation/allocator-output/flow.csv'),'mass_receipt':pin(O/'mass-receipt.json'),'physical_admission':False,'native_steps':0})
 save(O/'execution.json',{'exit_code':0,'wall_s':time.time()-start,'producer':pin(__file__),'unchanged_numerical_core_adapter':pin(module_path),'physical_admission':False,'native_steps':0})
 print('INDEPENDENT_ADMISSION_PASS',time.time()-start,'target_max_relative_error',maximum,flush=True)
 print('receipt_status',mass.get('status'),flush=True)
except Exception as e:
 save(O/'execution.json',{'exit_code':1,'wall_s':time.time()-start,'error':str(e),'traceback':traceback.format_exc(),'physical_admission':False,'native_steps':0});raise
