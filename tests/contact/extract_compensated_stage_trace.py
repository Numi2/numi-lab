#!/usr/bin/env python3
"""Extract one named capture from a retained rich Newton trace."""
import json
import pathlib
import sys

trace, prefix = sys.argv[1:]
found = []
for line in pathlib.Path(trace).read_text().splitlines():
    if line.startswith("fem_newton_iterate="):
        item = json.loads(line.split("=", 1)[1])
        if item["iteration"] == 3 and item["stage"] == "final_line_search" and item["state_arenas"]:
            found.append(item)
assert len(found) == 1 and found[0]["alpha"] == [1]
for key in ("working_nodes", "correction"):
    pathlib.Path(prefix + "-" + key + ".bin").write_bytes(bytes.fromhex(found[0]["state_arenas"][key]))
