#!/usr/bin/env python3
"""Measure the best possible stored FP32 velocity under each retained wrench."""
import hashlib,json,pathlib,struct
root=pathlib.Path(__file__).resolve().parents[2]
e=root/'docs/evidence/midpoint-first-failure-20261109'
names=['slider-primary-h15625-001','slider-near-h15625-001','slider-primary-h078125-001','slider-near-h078125-001']
f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
result=[]
for name in names:
 path=e/(name+'.json');d=json.loads(path.read_text());steps=[s for s in d['steps'] if s['accepted']]
 velocity=d['initial_slider_velocity_mps'];mass=d['slider_mass_kg'];h=d['h'];gravity=f32(-9.81)
 impossible=0;nearest=0;worst=0.;worst_ratio=0.
 for s in steps:
  ideal=velocity+h*gravity+s['owner_wrench_impulse']/mass
  best=f32(ideal);best_error=abs((best-ideal)*mass)
  bound=max(1e-12,1e-5*abs(s['fem_impulse']))
  impossible+=best_error>bound;nearest+=s['v']==best
  worst=max(worst,best_error);worst_ratio=max(worst_ratio,best_error/bound)
  velocity=s['v']
 result.append({'trace':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
  'accepted_steps':len(steps),'steps_when_nearest_float_is_insufficient':impossible,
  'steps_when_published_velocity_is_nearest':nearest,'largest_best_possible_impulse_error_Ns':worst,
  'largest_best_possible_gate_ratio':worst_ratio})
print(json.dumps({'schema':'numi.slider-float-resolution.v1','method':'For each accepted step, start from the stored velocity, use the measured body-wrench impulse and gravity, and round the resulting ideal velocity to the nearest binary32. Compare the resulting best possible impulse balance with the predeclared per-step gate. This does not assess unobserved state or prove the nonlinear rejection cause.','traces':result},indent=2,sort_keys=True))
