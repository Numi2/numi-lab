"""Replay a retained failed sweep; compare signed distance to an FP64 OBB model."""
from pathlib import Path
import json,math
import numpy as np
from metalrobo.crow_environment import execute,write,sha
root=Path.cwd();source=root/'.numi/runs/american-crow/ccd-failure-sweep-01'
assert json.loads((source/'summary-v2.json').read_text())['status']=='expected_failure_captured'
out=root/'.numi/runs/american-crow/ccd-failure-sweep-analysis-01';out.mkdir(parents=True,exist_ok=False)
build=root/'build/crow-ccd-failure-sweep-build-01'
command=[str(build/'metalrobo_ccd_sweep_probe'),str(build/'probe.metallib'),str(source/'failure.ccdsweep')]
write(out/'plan.json',{
 'question':'Do the captured projections/events reproduce the failed candidate, and is the underlying sphere-box signed distance accurate?',
 'requirements':'At least one event-cursor replay must reproduce every byte of the stored candidate. Uniform257-point query errors versus independent FP64 OBB distance<=8um and all native distance queries succeed; exact repeat of the complete result stream.',
 'interpretation':'Only byte-exact candidate replay admits this post-submission snapshot as the failing sweep. Report CA flags/iterations and event counters without treating a passing query as a complete CCD repair. A sampled clear path is certified only if a conservative between-sample geometric bound also clears tolerance.',
 'budget':'Two isolated native probe executions, timeout60s each, fixed retained input; no trajectory/learning execution or parameter tuning; raw<=128KiB.',
 'input_sha256':sha(source/'failure.ccdsweep'),'probe_sha256':sha(build/'metalrobo_ccd_sweep_probe'),
 'shader_sha256':sha(build/'probe.metallib'),'source_sha256':sha(root/'apps/ccd_sweep_probe.metal'),
 'reader_sha256':sha(Path(__file__))})
try:
 records=[]
 for name in ('replay','repeat'):
  records.append(execute(out,name,command+[str(out/(name+'.bin'))],root,60))
 data=(out/'replay.bin').read_bytes();assert data==(out/'repeat.bin').read_bytes()
 v=records[0];assert v==records[1]
 assert len(data)==128+257*16
 trace=np.frombuffer(data,dtype='<f4',offset=128).reshape(257,4).astype(float)
 assert np.isfinite(trace).all() and np.all(trace[:,2]==0) and np.all(trace[:,3]==1)
 assert any(v['stored_record_matches']), 'Post-submission sweep does not reproduce stored candidate'
 f32=lambda x:np.asarray(x,dtype=np.float32).astype(float)
 box,sphere=v['shapes'] if v['shapes'][0]['type']==2 else v['shapes'][::-1]
 h=f32(box['dimensions'])[:3];r=f32(sphere['dimensions'])[0]
 q0=f32(box['start']['rotation']);q1=f32(box['end']['rotation'])
 if q0@q1<0:q1=-q1
 b0=f32(box['start']['center_radius'])[:3];b1=f32(box['end']['center_radius'])[:3]
 s0=f32(sphere['start']['center_radius'])[:3];s1=f32(sphere['end']['center_radius'])[:3]
 def rotate(q,x):return x+2*np.cross(q[:3],np.cross(q[:3],x)+q[3]*x)
 def distance(alpha):
  q=(1-alpha)*q0+alpha*q1;q/=np.linalg.norm(q);q[:3]*=-1
  local=rotate(q,(1-alpha)*(s0-b0)+alpha*(s1-b1))
  edge=np.abs(local)-h
  return float(np.linalg.norm(np.maximum(edge,0))+min(float(np.max(edge)),0)-r)
 expected=np.asarray([distance(a) for a in trace[:,0]])
 error=np.abs(trace[:,1]-expected);assert error.max()<=8e-6
 # A Hausdorff bound for the independent FP64 geometric path. For
 # nlerp z=q0+alpha*dq, ||d(z/||z||)/dalpha||<=||dq||/min||z||;
 # rotation angular speed is twice quaternion tangent speed.
 dq=q1-q0;denominator=float(dq@dq)
 minimizing_alpha=float(np.clip(-(q0@dq)/denominator,0,1)) if denominator else 0.
 minimum_quaternion_norm=float(np.linalg.norm(q0+minimizing_alpha*dq))
 assert minimum_quaternion_norm>0
 angular_bound=2*float(np.linalg.norm(dq))/minimum_quaternion_norm
 motion_bound=float(np.linalg.norm(b1-b0)+np.linalg.norm(s1-s0)+np.linalg.norm(h)*angular_bound)
 minimum_lower_bound=float(expected.min()-motion_bound/512)
 tolerance=float(f32(v['ccd_parameters'])[1])
 result={'status':'admitted_exact_candidate_replay','stored_record_matches_event_cursors':v['stored_record_matches'],
  'native_repeat_exact':True,'queries':257,'query_FP64_max_abs_error_m':float(error.max()),
  'sample_minimum_distance_m':float(expected.min()),'sample_maximum_distance_m':float(expected.max()),
  'sample_minimum_alpha':float(trace[np.argmin(expected),0]),'ccd_tolerance_m':tolerance,
  'normalized_sweep_motion_bound_m':motion_bound,'nlerp_angular_speed_bound_rad_per_alpha':angular_bound,
  'whole_sweep_distance_lower_bound_m':minimum_lower_bound,
  'whole_sweep_FP64_geometric_clearance_certified':bool(minimum_lower_bound>tolerance),
  'stored_record':v['stored_record'],'events':v['events'],'status_counters':v['status_counters'],
  'maximum_CA_iterations':v['maximum_CA_iterations'],
  'scope':'Same failed candidate and sampled distance accuracy. Whole-sweep clearance bounds concern the independent FP64 geometric path, not a formal roundoff bound for every native FP32 operation. No physics repair, learning success or biological qualification.'}
 write(out/'analysis.json',result)
 print(json.dumps(result,indent=2))
except Exception as exc:
 write(out/'failure.json',{'type':type(exc).__name__,'error':str(exc)});raise
