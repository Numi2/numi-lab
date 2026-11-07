#!/usr/bin/env python3
import collections
import csv
import json

data = collections.defaultdict(dict)
with open("snapshots.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        data[row["phase"], row["array"]][int(row["index"])] = int(row["bits"])

changes = {}
for name, stride in (("depth", 144), ("identity", 576), ("validity", 144), ("camera", 16), ("track", 48)):
    before, after = data["after", name], data["continued", name]
    assert len(before) == 2 * stride and len(after) == 2 * stride
    changes[name] = [sum(before[i] != after[i] for i in range(env * stride, (env + 1) * stride)) for env in range(2)]
prior, accepted = data["prior", "metadata"], data["accepted", "metadata"]
assert len(prior) == len(accepted) == 32
changes["accepted_metadata"] = [sum(prior[env * 16 + i] != accepted[env * 16 + i] for i in range(16)) for env in range(2)]
print(json.dumps(changes, indent=2, sort_keys=True))

