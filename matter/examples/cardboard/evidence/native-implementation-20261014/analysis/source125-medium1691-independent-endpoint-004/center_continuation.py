#!/usr/bin/env python3
import datetime, importlib.util, json, pathlib, hashlib
import numpy as np
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
PKG=BASE/'analysis/source125-medium1691-independent-endpoint-004'
D=importlib.util.spec_from_file_location('diagnose',PKG/'diagnose.py'); diag=importlib.util.module_from_spec(D); D.loader.exec_module(diag)
a=diag.load(diag.SCRIPT003,'audit003_center'); ref=diag.load(diag.REFPATH,'ref15_center')
records=[json.loads(line) for line in diag.CAP.read_text().splitlines() if line.strip()]
h=records[0]; tets=sorted((r for r in records if r.get('record')=='tetrahedron' and r.get('environment')==0),key=lambda r:int(r['tet_id']))
invrest=np.asarray([a.f32(t['inverse_rest_rows_per_m']) for t in tets],dtype=np.float64)
frames=np.asarray([a.frame_from_xyzw(t['material_frame_quaternion_xyzw']) for t in tets],dtype=np.float32)
active=np.asarray([int(a.f32([t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]])[0])==1 for t in tets])
mat={int(x['index']):x for x in h['materials']}; t=tets[1691]; m=mat[1]
ptab=a.f32(h['parameters_by_environment'][0]); off=int(m['parameter_offset']); p=a.f32(ptab[off:off+len(m['parameter_names'])]); old=a.f32(t['center']['accepted_state'])
def nodes(stem): return np.fromfile(diag.SWEEP/'result'/f'{stem}-nodes.bin',dtype='<f4').reshape(2,882,5,4)[0]
Fc_hi,Fc_lo=a.source_deformation_words(nodes('sample-0-0'),tets,invrest,frames,active)
Fp_hi,Fp_lo=a.source_deformation_words(nodes('sample-6-0'),tets,invrest,frames,active)
Fc=Fc_hi[1691].astype(np.float64)+Fc_lo[1691].astype(np.float64)
Fp=Fp_hi[1691].astype(np.float64)+Fp_lo[1691].astype(np.float64)
zc,ic,rc=ref.project(p[None],Fc[None],old[None],maxiter=128)
anchor=diag.solve_checked(ref,p,Fc,old,zc[0],max_updates=32,max_backtracks=64)
assert anchor['ok']
results=[]
for steps in (1,2,4,8):
 z=anchor['z'].copy(); path=[]; ok=True
 for k in range(1,steps+1):
  tau=k/steps; Fi=Fc+(Fp-Fc)*tau
  r=diag.solve_checked(ref,p,Fi,old,z,max_updates=32,max_backtracks=64)
  path.append({'step':k,'fraction':tau,'result':{q:v for q,v in r.items() if q not in ('z',)},'state':r['z'].tolist()})
  if not r['ok']:
   ok=False; break
  z=r['z']
 results.append({'steps_requested':steps,'steps_completed':len(path),'ok':ok,'path':path,
                 'final_state':z.tolist() if ok else None,
                 'final_branch':diag.branch(ref,p,old,z) if ok else None})
# Compare all successful endpoint solutions to the earlier minus-anchor solution.
prior=json.loads((PKG/'result.json').read_text())
prior_z=np.asarray(prior['recovered_endpoint']['state'],dtype=np.float64)
for row in results:
 if row['ok']:
  state=np.asarray(row['final_state'])
  row['vs_minus_anchor_success_linf']=float(np.max(np.abs(state-prior_z)))
  row['vs_minus_anchor_state_scaled_linf']=float(np.max(np.abs(state-prior_z)/np.maximum(np.maximum(np.abs(old),np.abs(state)),p[26])))
report={'schema':'numi.cardboard.medium1691-center-to-plus-continuation.v1','status':'diagnostic_only','tet_id':1691,
 'bindings':{'diagnose_script_sha256':diag.sha(PKG/'diagnose.py'),'center_nodes_sha256':diag.sha(diag.SWEEP/'result/sample-0-0-nodes.bin'),
  'plus_nodes_sha256':diag.sha(diag.SWEEP/'result/sample-6-0-nodes.bin'),'accepted_history_source':diag.sha(diag.CAP)},
 'anchor':{'ok':anchor['ok'],'iterations':len(anchor['history']),'scaled_residual':anchor.get('scaled_residual'),'scaled_correction':anchor.get('scaled_correction'),'branch':diag.branch(ref,p,old,anchor['z'])},
 'attempts':results,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(PKG/'center-continuation.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'anchor':report['anchor'],'attempts':[{'steps':x['steps_requested'],'ok':x['ok'],'steps_completed':x['steps_completed'],'final_branch':x['final_branch'],'vs_minus_anchor_state_scaled_linf':x.get('vs_minus_anchor_state_scaled_linf')} for x in results]},indent=2,allow_nan=False))
