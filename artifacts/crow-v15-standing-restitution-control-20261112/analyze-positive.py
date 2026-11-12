import contextlib,io,runpy,json,collections
from pathlib import Path
with contextlib.redirect_stdout(io.StringIO()):d=runpy.run_path('artifacts/crow-v15-standing-restitution-control-20261112/analyze.py')
A=d['A']; c=d['v']['control'];z=d['v']['zero']; b=d['read'](A/'candidate-20261127-rebound-boost')
res={'zero_control_substep_differences':sum(c[1][x]!=z[1][x] for x in c[1]),'boost_control_endpoint_differences_by_lane':{},'boost_control_substep_differences_by_lane':{},'boost_unloaded_substeps_0_to_2s':{},'boost_unloaded_substeps_2_to_3s':{},'boost_toe_speed_cap_substeps':dict(b[2]),'boost_errors':b[3]}
for e in range(8):
 res['boost_control_endpoint_differences_by_lane'][e]=sum(c[0][x]!=b[0][x] for x in c[0] if x[0]==e)
 res['boost_control_substep_differences_by_lane'][e]=sum(c[1][x]!=b[1][x] for x in c[1] if x[0]==e)
 for key,lo,hi in [('boost_unloaded_substeps_0_to_2s',1,100),('boost_unloaded_substeps_2_to_3s',101,150)]:
  res[key][e]=sum(float(b[1][e,t,k][4])<=1e-7 and float(b[1][e,t,k][5])<=1e-7 for t in range(lo,hi+1) for k in range(16))
ep=json.loads((A/'candidate-20261127-rebound-boost/run.stdout').read_text())['first_episode_navigation'];res['boost_episode']=ep
(A/'positive-analysis.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
