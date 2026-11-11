from pathlib import Path
import gzip,json
a=Path('artifacts/crow-v15-start-perturbation-controls-20261111')
sources={'double':Path('artifacts/crow-v15-hip-roll-walk-launch-controls-20261111/gait-080-20261126'),'single':a/'single-20261126','none':a/'none-20261126'}
report={}
for name,p in sources.items():
 counts=[[0,0,0] for _ in range(8)]
 maximum=[[0.,0.,0.] for _ in range(8)]
 coverage=[[0,0,0] for _ in range(8)]
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_TOE_V_SUBSTEP '):
    x=line.split();t,k,e=map(int,x[1:4])
    if t>301:continue
    phase=0 if t<101 else 1 if t<201 else 2
    v=max(abs(float(z)) for z in x[4:]);coverage[e][phase]+=1;maximum[e][phase]=max(maximum[e][phase],v);counts[e][phase]+=v>=11.99
 report[name]={'phase_ranges_inclusive':[[1,100],[101,200],[201,301]],'cap_samples_by_lane_and_phase':counts,'max_sampled_toe_speed_rad_s_by_lane_and_phase':maximum,'coverage_by_lane_and_phase':coverage,'scope':'Substep endpoint velocities; these do not bound intra-substep peaks.'}
(a/'toe-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
for k,v in report.items():print(k,'cap samples per phase',list(map(sum,zip(*v['cap_samples_by_lane_and_phase']))),'coverage ranges',[[min(x[i] for x in v['coverage_by_lane_and_phase']),max(x[i] for x in v['coverage_by_lane_and_phase'])] for i in range(3)])
