from pathlib import Path
import importlib.util, numpy as np, json, time
p=Path('/Users/home/cardboard-implementation-20261014/analysis/local-projection-scaled-correction-082/replay_scaled_policy.py')
spec=importlib.util.spec_from_file_location('rp',p);rp=importlib.util.module_from_spec(spec);spec.loader.exec_module(rp)
rows=rp.material_rows(rp.N0_CAPTURE)[1]
npz=np.load(rp.N0_ARRAYS)
strict=npz['referenceProjectedT']
allout=[]
for tol in [1e-8,5e-9,2e-9,1e-9]:
 t=time.time(); rp.CORRECTION_TOL=tol
 roots=[]; infos=[]; Ts=[]
 for row in rows:
  z,info=rp.project_scaled(row['p'],row['F'],row['old'],row['candidate'])
  roots.append(z);infos.append(info)
  Ts.append(rp.ref.tangent(row['p'],row['F'],row['old'],z,row['direction']))
 T=np.asarray(Ts); diff=T-strict
 rowd=np.linalg.norm(diff,axis=(1,2))/np.maximum(np.linalg.norm(strict,axis=(1,2)),1e-300)
 statuses={}
 for i in infos: statuses[i['status']]=statuses.get(i['status'],0)+1
 allout.append({'correction_tolerance':tol,'status_counts':statuses,
   'mean_iterations':float(np.mean([i['iterations'] for i in infos])),
   'max_iterations':int(max(i['iterations'] for i in infos)),
   'max_authored_residual_fraction':float(max(i.get('residual_inf_scaled',0)/max(i.get('authored_residual_threshold',1e-300),1e-300) for i in infos)),
   'max_correction_fraction':float(max(i.get('scaled_correction_inf',0)/tol for i in infos)),
   'global_l2_relative_to_strict':float(np.linalg.norm(diff)/max(np.linalg.norm(strict),1e-300)),
   'max_per_tet_l2_relative_to_strict':float(rowd.max()),
   'worst_tet_ids':[{'array_index':int(i),'tet_id':rows[int(i)]['tet_id'],'rel':float(rowd[i]),'iterations':infos[int(i)]['iterations'],'corr':infos[int(i)].get('scaled_correction_inf')} for i in np.argsort(rowd)[-6:][::-1]],
   'seconds':time.time()-t})
 print(json.dumps(allout[-1],indent=2))
out=Path(__file__).parent/'correction_tolerance_sweep.json';out.write_text(json.dumps(allout,indent=2)+'\n')
