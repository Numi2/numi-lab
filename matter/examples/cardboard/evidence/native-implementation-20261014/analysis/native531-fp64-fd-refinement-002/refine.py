#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
BASE=Q/'analysis/native531-strict-fp64-replay-001/layers'
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
CASE=Q/'solver/control17-root17-newton34-replay-531'
OUT=Q/'analysis/native531-fp64-fd-refinement-002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((BASE/'manifest.json').read_text())
for n,h in m['files_sha256'].items(): assert sha(BASE/n)==h,n
assert sha(BASE/'manifest.json')=='1db4342cdf9b866cd8280209a86cdc7cb70cd712ddb8ea68b2d42b0eb89253da'
oldr=json.loads((BASE/'report.json').read_text())
assert oldr['fp64_directional_sweep']['rows_with_two_adjacent_declared_grid_points_at_1e_6']==1718
with np.load(BASE/'arrays.npz') as z:
    unresolved=np.flatnonzero(~z['fp64_resolved_adjacent_pair']).astype(int).tolist()
assert unresolved==[136,498,499,510,511,650,1012,1362,1363,1523],unresolved
sys.path.insert(0,str(PREP))
S=Q/'snapshots/source-contact-damped-memory-198-001/matter/tools'
os.environ['PYTHONPATH']=str(S)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import strict_layers
powers=list(range(10,25))
start=time.monotonic()
report=strict_layers.audit(CASE,OUT/'layers',environment=0,epsilon_powers=powers)
with np.load(OUT/'layers/arrays.npz') as z:
    remaining=np.flatnonzero(~z['fp64_resolved_adjacent_pair']).astype(int).tolist()
    rows=[]
    for i in unresolved:
        grid=[]
        for power in powers:
            grid.append({'power':power,'epsilon':2.0**(-power),'smooth':bool(z[f'e{power}_smooth'][i]),'relative_l2':float(z[f'e{power}_relative_error'][i]),'plus_branch_preserving':bool(z[f'e{power}_plus_branch_preserving'][i]),'minus_branch_preserving':bool(z[f'e{power}_minus_branch_preserving'][i])})
        rows.append({'tet':i,'material_index':int(z['material'][i]),'grid':grid,'adjacent_pair_resolved':bool(z['fp64_resolved_adjacent_pair'][i])})
pred=json.loads((CASE/'prediction.json').read_text())
capture=Path(pred['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH'])
summary={'schema':'numi.cardboard.native531-fp64-fd-refinement.v1','classification':'CPU-only reference epsilon refinement; no native action/noise floor/solver qualification','parent_report_sha256':sha(BASE/'report.json'),'parent_arrays_sha256':sha(BASE/'arrays.npz'),'parent_layer_manifest_sha256':sha(BASE/'manifest.json'),'prediction_sha256':sha(CASE/'prediction.json'),'capture_sha256':sha(capture),'strict_layer_script_sha256':sha(PREP/'strict_layers.py'),'unresolved_input_rows':unresolved,'grid_powers':powers,'total_paper_rows':1728,'resolved_adjacent_pair_count':report['fp64_directional_sweep']['rows_with_two_adjacent_declared_grid_points_at_1e_6'],'remaining_unresolved_rows':remaining,'per_row_results':rows,'interpretation':'Every original unresolved row is retained. Adjacent passes refer only to compensated-FP64 ideal endpoint finite differences at fixed accepted history. This is not a measured native FP32 noise floor or native branch trace.','wall_seconds':time.monotonic()-start}
(OUT/'unresolved-rows.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
manifest={'schema':'numi.cardboard.native531-fp64-fd-refinement-files.v1','parent':str(BASE),'inputs_sha256':{str(BASE/n):sha(BASE/n) for n in ('manifest.json','report.json','arrays.npz')},'case_inputs_sha256':{str(CASE/n):sha(CASE/n) for n in ('prediction.json','receipt.json','stderr.log','output/manifest.json')},'capture_sha256':sha(capture),'tool_sha256':{str(PREP/'strict_layers.py'):sha(PREP/'strict_layers.py'),str(OUT/'refine.py'):sha(OUT/'refine.py'),str(OUT/'unresolved-rows.json'):sha(OUT/'unresolved-rows.json')}}
manifest['files_sha256']={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'summary_sha256':sha(OUT/'unresolved-rows.json'),'manifest_sha256':sha(OUT/'manifest.json'),'rows_resolved':len(unresolved)-len(remaining),'remaining':remaining,'all_rows_adjacent':report['fp64_directional_sweep']['all_rows_have_resolved_pair'],'wall_seconds':summary['wall_seconds']},indent=2))
