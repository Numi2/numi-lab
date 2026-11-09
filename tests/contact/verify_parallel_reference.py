#!/usr/bin/env python3
"""Verify byte-for-byte parity of completed independent-reference trials."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/"docs/evidence"
old=root/"midpoint-spatial-cost-dynamic-20261108"
new=root/"midpoint-first-failure-20261109"
result={}
for n,h in ((2,"025"),(8,"025"),(32,"050")):
    a=(old/f"rk4-n{n}-h{h}.json").read_bytes()
    b=(new/f"fp64-parallel-n{n}.json").read_bytes()
    result[f"n{n}"]={"identical":a==b,"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
print(json.dumps(result,indent=2,sort_keys=True))
if not all(v["identical"] for v in result.values()):raise SystemExit(1)
