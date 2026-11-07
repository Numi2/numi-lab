#!/usr/bin/env python3
"""Independent transaction and normal-response gates for the native probe."""
import json
import subprocess
import sys

p=subprocess.run([sys.argv[1],sys.argv[2]],check=True,text=True,capture_output=True)
rows=json.loads(p.stdout)["cases"]
assert len(rows)==24
for r in rows:
    assert r["replay_identical"] and r["state_restored"] and r["rejected_contact_cache_empty"]
    assert r["active_contacts"]==r["contacts"]
    assert r["distributed"]==(r["contacts"]==257)
    if r["initial_vn"]>=0:
        assert r["accepted"] and r["status"]==0
        assert abs(r["final_vn"]-r["initial_vn"])<1e-5
    elif r["contacts"]==257:
        if r["accepted"]:
            assert abs(r["final_vn"]) <= .0101 and r["normal_residual"]<=.0101
        else:
            assert r["status"]==11
    else:
        assert r["accepted"] and r["status"]==0
print(p.stdout)
