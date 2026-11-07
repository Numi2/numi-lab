#!/usr/bin/env python3
"""Independent transaction and normal-response gates for the native probe."""
import json
import subprocess
import sys

p=subprocess.run([sys.argv[1],sys.argv[2]],check=True,text=True,capture_output=True)
rows=json.loads(p.stdout)["cases"]
assert len(rows)==72
for r in rows:
    assert r["replay_identical"] and r["state_restored"] and r["rejected_contact_cache_empty"]
    assert r["active_contacts"]==r["contacts"]
    assert r["distributed"]==(r["contacts"]==257)
    if r["initial_vn"]>=0:
        assert r["accepted"] and r["status"]==0
        assert abs(r["final_vn"]-r["initial_vn"])<1e-5
        assert abs(r["normal_impulse_ns"])<1e-5
        assert abs(r["final_vt"]-r["initial_vt"])<1e-5
    elif r["contacts"]==257:
        if r["accepted"]:
            tolerance=.0002 if r["mixed_normals"] else .0101
            assert abs(r["final_vn"]) <= tolerance and r["normal_residual"]<=tolerance
            expected=.03 if r["mixed_normals"] else 1.
            assert abs(r["normal_impulse_ns"]-expected)<=tolerance*3
            assert r["momentum_impulse_error_ns"]<1e-5
            if r["friction"]>0:
                assert abs(r["final_vt"]-.5) <= .03
                assert abs(r["tangent_impulse_magnitude_ns"]-.5) <= .03
        else:
            assert r["status"]==11
    else:
        assert r["accepted"] and r["status"]==0
print(p.stdout)

if "--require-production-completion" in sys.argv:
    assert all(r["accepted"] for r in rows if r["contacts"]==257
               and r["iterations"]==2 and r["initial_vn"]<0), "default-budget impacts did not all complete"
