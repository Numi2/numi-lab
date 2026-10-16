#!/usr/bin/env python3
"""Deterministically describe checker cases without compiling or dispatching Metal."""
import json
from pathlib import Path

OUT = Path(__file__).with_name("gpu-case-inventory.json")
SIZES = [1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1025]

def random_count(n, seed=0x12345678):
    state = seed or 1
    count = 0
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xffffffff
        count += (state >> 27) < 3
    return count

rows = []
for n in SIZES:
    cap = max(1, n // 2)
    tail = sum(i % 256 >= 248 for i in range(n))
    rows.extend([
        {"name": ("all_fit_" if n == 1 else "all_overflow_") + str(n), "environments": 1, "pairs_per_environment": n, "capacity_per_environment": cap, "selected_per_environment": [n], "health": ["healthy"]},
        {"name": f"none_{n}", "environments": 1, "pairs_per_environment": n, "capacity_per_environment": cap, "selected_per_environment": [0], "health": ["healthy"]},
        {"name": f"block_tail_{n}", "environments": 1, "pairs_per_environment": n, "capacity_per_environment": n, "selected_per_environment": [tail], "health": ["healthy"]},
    ])
for name, selected in [("capacity_minus_one",127),("capacity_exact",128),("capacity_plus_one",129)]:
    rows.append({"name":name,"environments":1,"pairs_per_environment":257,"capacity_per_environment":128,"selected_per_environment":[selected],"health":["healthy"]})
rows.extend([
    {"name":"alternating_multi_block","environments":1,"pairs_per_environment":4097,"capacity_per_environment":2048,"selected_per_environment":[2049],"health":["healthy"]},
    {"name":"deterministic_random","environments":1,"pairs_per_environment":8193,"capacity_per_environment":8193,"selected_per_environment":[random_count(8193)],"health":["healthy"]},
    {"name":"two_environment_healthy_overflow_and_prefailed_partial","environments":2,"pairs_per_environment":513,"capacity_per_environment":64,"selected_per_environment":[513,3],"health":["healthy","prefailed_partial_flags"]},
    {"name":"shipping_pair_domain_no_overflow","environments":1,"pairs_per_environment":6335568,"capacity_per_environment":65536,"selected_per_environment":[(6335568+96)//97],"health":["healthy"],"selection":"pair ids 0,97,194,... < 6,335,568"},
])
obj = {
  "schema":"numi.cardboard.sparse-private-scan-gpu-case-inventory.v1",
  "source_checker":"source/cardboard_sparse_private_contact_scan_native_check.mm",
  "native_case_count":len(rows),
  "total_environment_pair_slots":sum(r["environments"]*r["pairs_per_environment"] for r in rows),
  "cases":rows,
  "note":"Inventory derived from the checker case constructors. No GPU execution is represented by this file."
}
OUT.write_text(json.dumps(obj, sort_keys=True, indent=2)+"\n")
print(json.dumps({k:obj[k] for k in ("native_case_count","total_environment_pair_slots")},sort_keys=True))
