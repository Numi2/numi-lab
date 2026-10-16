#!/usr/bin/env python3
"""Hash the actual source176 compiler dependency closure on macmini."""
from __future__ import annotations
import hashlib, json, shlex, sys
from pathlib import Path

depfile = Path(sys.argv[1])
source_root = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3])
text = depfile.read_text().replace("\\\n", " ")
_, rhs = text.split(":", 1)
tokens = shlex.split(rhs)
source_rows = []
external = []
for raw in tokens:
    p = Path(raw)
    try:
        rel = p.resolve().relative_to(source_root)
    except (OSError, ValueError):
        external.append(raw)
        continue
    data = p.read_bytes()
    source_rows.append({"relative_path": rel.as_posix(),
                        "byte_count": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
source_rows.sort(key=lambda row: row["relative_path"])
payload = {"schema": "numi.cardboard.source176_metal_dependency_hashes.v1",
           "source_root": str(source_root),
           "dependencies": source_rows,
           "external_dependencies": sorted(external)}
out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
print(json.dumps({"dependency_count": len(source_rows),
                  "external_count": len(external),
                  "output": str(out)}, sort_keys=True))
