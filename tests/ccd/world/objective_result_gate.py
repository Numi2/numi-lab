#!/usr/bin/env python3
"""Check retained paired-row outputs; status is not a force-accuracy guarantee."""
import json,sys
d=json.load(open(sys.argv[1]))["cases"]
assert len(d)==72
assert all(r["replay_identical"] and r["state_restored"] and r["rejected_contact_cache_empty"] for r in d)
assert sum(r["accepted"] for r in d)==71
for r in d:
 if r["initial_vn"]>=0:
  assert r["accepted"] and abs(r["normal_impulse_ns"])<1.e-5
  assert abs(r["final_vn"]-r["initial_vn"])<1.e-5
  assert abs(r["final_vt"]-r["initial_vt"])<1.e-5
 elif r["contacts"]==257:
  assert r["accepted"]
  tol=.0002 if r["mixed_normals"] else .0101
  expected=.03 if r["mixed_normals"] else 1.
  assert abs(r["final_vn"])<=tol and r["normal_residual"]<=tol
  assert abs(r["normal_impulse_ns"]-expected)<=3*tol
  assert r["momentum_impulse_error_ns"]<1.e-5
  if r["friction"]>0:
   assert abs(r["final_vt"]-.5)<=.03
   assert abs(r["tangent_impulse_magnitude_ns"]-.5)<=.03
 elif r["iterations"]==1 and not r["mixed_normals"] and r["friction"]>0:
  assert not r["accepted"] and r["status"]==11
 else:
  assert r["accepted"] and r["status"]==0
print("objective-priority physical matrix gate passed with one previously unsafe case rejected")
