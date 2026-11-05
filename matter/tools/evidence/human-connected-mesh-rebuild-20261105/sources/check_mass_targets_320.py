from pathlib import Path
import json,hashlib,numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
C=W/'joint-boundary-bone-admitted-320/compiler'
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':h}
run=json.loads((C/'mass-allocation/run-manifest.json').read_text())
receipt=C/'mass-admission/mass-receipt.json'
assert json.loads(receipt.read_text())['status']=='PASS_SOURCE_ONLY_REPLAY'
flowpath=C/'mass-allocation/allocator-output/flow.csv'
flow=np.loadtxt(flowpath,delimiter=',',skiprows=1,usecols=(0,2))
actual=np.bincount(flow[:,0].astype(np.int64),weights=flow[:,1]);del flow
targetpath=Path(run['prescribed_child_mass']['path'])
assert pin(targetpath)==run['prescribed_child_mass']
target=np.loadtxt(targetpath,delimiter=',',skiprows=1)
rows=target[:,0].astype(np.int64);values=target[:,1]
assert np.array_equal(rows.astype(float),target[:,0]) and np.all(np.diff(rows)>0)
assert np.all(np.isfinite(actual)) and np.all(np.isfinite(values)&(values>0))
rel=np.abs(actual[rows]-values)/values;maximum=float(rel.max())
assert maximum<=1e-10+64*np.finfo(float).eps,(maximum,int(rows[rel.argmax()]))
result={'status':'PASS_INDEPENDENT_RECORDED_FLOW_TARGET_PARITY','prescribed_cells':len(rows),'maximum_relative_mass_difference':maximum,'maximum_absolute_mass_difference_kg':float(np.abs(actual[rows]-values).max()),'target':run['prescribed_child_mass'],'flow':pin(flowpath),'mass_receipt':pin(receipt),'physical_admission':False,'native_steps':0}
(C/'mass-admission/prescribed-parity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
