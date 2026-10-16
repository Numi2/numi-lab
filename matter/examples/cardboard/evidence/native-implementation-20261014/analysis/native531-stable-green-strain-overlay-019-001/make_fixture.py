import sys,importlib.util,json,hashlib
from pathlib import Path
import numpy as np
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PK=Q/'analysis/native531-stable-green-strain-overlay-019-001'
oldpath=Q/'snapshots/source-cohesive-quadrature-area-205/matter/tools/cardboard_bilinear_fp64_reference.py'
stablepath=Q/'analysis/native531-stable-green-strain-reference-017-001/reference/cardboard_bilinear_fp64_reference.py'
tools=Q/'snapshots/source-cohesive-quadrature-area-205/matter/tools'
sys.path.insert(0,str(tools))
import cardboard_compensated_fp64_reference as strict
spec=importlib.util.spec_from_file_location('stable_reference_test',stablepath);stable=importlib.util.module_from_spec(spec);spec.loader.exec_module(stable)
import cardboard_bilinear_fp64_reference as old

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def branch(base,p,oldstate,z):
 b=base.paper_branch(base,p,oldstate,z);dl=b['plastic_multiplier_increment'];y=b['yield_function_residual_value'];den=np.sqrt(dl*dl+y*y+p[:,26]**2)+dl+y
 return np.stack((np.sign(dl),np.sign(y),den>1e-14),axis=1)
def run(base,p,F,oldstate,h):
 engine=strict.Reference(base)
 z,it,rn,cn,ok,why,prov=engine.project_with_metrics(p,F,oldstate)
 if not ok[0]: return {'center_ok':False,'why':why[0],'rn':float(rn[0]),'cn':float(cn[0])}
 tan=strict.tangent(base,p,F,oldstate,z,h);center=base.stress(p,F,z);b0=branch(base,p,oldstate,z)
 result={'center_ok':True,'center_iterations':int(it[0]),'center_rn':float(rn[0]),'center_cn':float(cn[0]),'center_branch':b0[0].tolist(),'center_state':z[0].tolist(),'center_stress':center[0].tolist(),'tangent':tan[0].tolist(),'directions':[]}
 for power in [33,34]:
  eps=2.**-power; endpoints=[]
  for sign in [1.,-1.]:
   ff=F+sign*eps*h;zz,ii,rr,cc,good,why,pr=engine.project_with_metrics(p,ff,oldstate)
   if not good[0]:endpoints.append({'valid':False,'why':why[0],'iterations':int(ii[0]),'rn':float(rr[0]),'cn':float(cc[0])});continue
   stress=base.stress(p,ff,zz);bb=branch(base,p,oldstate,zz)
   endpoints.append({'valid':True,'iterations':int(ii[0]),'rn':float(rr[0]),'cn':float(cc[0]),'branch':bb[0].tolist(),'state':zz[0].tolist(),'stress':stress[0].tolist()})
  if all(x['valid'] for x in endpoints):
   fd=(np.asarray(endpoints[0]['stress'])-np.asarray(endpoints[1]['stress']))/(2*eps)
   err=float(np.linalg.norm(fd-tan[0])/np.linalg.norm(tan[0]))
   same=all(np.array_equal(x['branch'],b0[0]) for x in endpoints)
  else:err=None;same=False
  result['directions'].append({'power':power,'epsilon':eps,'plus':endpoints[0],'minus':endpoints[1],'same_branch_both':same,'centered_relative_error':err})
 return result
arrpath=Q/'analysis/native531-strict-fp64-replay-001/layers/arrays.npz';npath=Q/'analysis/native531-fp64-ninebasis-003/arrays.npz';npz=np.load(arrpath);nine=np.load(npath);row=int(np.flatnonzero(nine['paper_tet_ids']==222)[0]);F=nine['center_F'][row:row+1];oldstate=nine['accepted_history'][row:row+1]
predpath=Q/'solver/control17-root17-newton34-replay-531/prediction.json';pred=json.loads(predpath.read_text());capture=Path(pred['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH']);sys.path.insert(0,str(Q/'analysis/native531-stable-green-strain-reference-017-001/reference'));import cardboard_cached_operator_audit as geo
header,nodes,tets,_=geo.capture_records(capture);tet=next(t for t in tets if int(t['environment'])==0 and int(t['tet_id'])==222);mat=next(m for m in header['materials'] if int(m['index'])==int(tet['material_index']));params=np.asarray(header['parameters_by_environment'][0],dtype=np.float64);off=int(mat['parameter_offset']);p=params[off:off+28][None,:];h=np.zeros((1,3,3));h.reshape(1,9)[0,6]=1.
oldres=run(old,p,F,oldstate,h);newres=run(stable,p,F,oldstate,h)
fixture={'schema':'numi.cardboard.stable-green-strain-tet222-col6-fixture.v1','provenance':{'source205_reference_sha256':sha(oldpath),'source203_reference_sha256':sha(Q/'snapshots/source-repeated-damped-fallback-203/matter/tools/cardboard_bilinear_fp64_reference.py'),'capture_sha256':sha(capture),'prediction_sha256':sha(predpath),'strict_layers_arrays_sha256':sha(arrpath),'ninebasis_arrays_sha256':sha(npath),'material_name':mat['name'],'tet':222,'basis_column':6},'input':{'F_center':F[0].tolist(),'accepted_history':oldstate[0].tolist(),'parameters':p[0].tolist(),'H':h[0].tolist(),'epsilon_powers':[33,34]},'legacy_source_result':oldres,'stable_source_result':newres,'gates':{'scaled_residual':2e-12,'dynamic_correction':2e-11,'same_center_branch':True,'adjacent_or_selected_pair_relative_fd':1e-6},'classification':'CPU-only focused reference regression. It does not certify native FP32 noise, material calibration, whole coupon, or production solver.'}
(PK/'tet222-col6-fixture.json').write_text(json.dumps(fixture,indent=2,allow_nan=False)+'\n')
print(json.dumps({'legacy':{k:v for k,v in oldres.items() if k not in ['center_state','center_stress','tangent']},'stable':{k:v for k,v in newres.items() if k not in ['center_state','center_stress','tangent']},'fixture_sha256':sha(PK/'tet222-col6-fixture.json')},indent=2))
