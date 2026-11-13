from pathlib import Path
import gzip,json,numpy as np,hashlib
A=Path(__file__).resolve().parent
with gzip.open(A/'moments.tsv.gz','rt') as f:a=np.loadtxt(f)
assert a.shape==(20160,10)
blocks=[a[i*4032:(i+1)*4032] for i in range(5)]
for i,b in enumerate(blocks):
 assert np.all(b[:,0]==i) and np.array_equal(b[:,1:4],blocks[0][:,1:4])
assert len({tuple(x) for x in blocks[0][:,1:4]})==4032
stderr=(A/'run.stderr').read_text().strip()
assert stderr=='baseline_max_abs_error 0 baseline_rms_error 0 n 96768',stderr
out={'n_states':4032,'baseline_component_count':96768,'max_abs_error':0,'description':'central difference per radian offset of each wing; differential = left +delta, right -delta; symmetric = both +delta','lanes':{}}
for lane in (6,7):
 mask=blocks[0][:,1]==lane
 row={}
 for name,p,m in [('differential',1,2),('symmetric',3,4)]:
  d=(blocks[p][mask,4:]-blocks[m][mask,4:])/.04
  row[name]={'mean_force_N_per_rad':d[:,:3].mean(axis=0).tolist(),'mean_moment_Nm_per_rad':d[:,3:].mean(axis=0).tolist(),'moment_min_Nm_per_rad':d[:,3:].min(axis=0).tolist(),'moment_max_Nm_per_rad':d[:,3:].max(axis=0).tolist(),'positive_yaw_fraction':float((d[:,5]>0).mean())}
 out['lanes'][str(lane)]=row
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
