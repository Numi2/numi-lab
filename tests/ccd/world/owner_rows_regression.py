#!/usr/bin/env python3
"""Verify all-path containment and motion-free controls, not response convergence."""
import json, subprocess, sys
p = subprocess.run([sys.argv[1], sys.argv[2]], check=True, capture_output=True, text=True)
rows = json.loads(p.stdout)["cases"]
assert len(rows) == 44
for r in rows:
    assert r["replay_identical"] and r["state_restored"] and r["rejected_contact_cache_empty"]
    if r["initial_vn"] < 0:
        assert not r["accepted"] and r["status"] == 11
        assert r["normal_residual"] > .00091749
        assert r["normal_impulse_ns"] is None
    else:
        assert r["accepted"] and r["status"] == 0
        assert r["normal_impulse_ns"] == 0 and r["tangent_impulse_magnitude_ns"] == 0
        assert abs(r["final_vn"] - r["initial_vn"]) < 1.e-6
        assert r["active_contacts"] == (0 if r["no_contact"] else r["contacts"])
print(p.stdout)
