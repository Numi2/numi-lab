# Source101 cached-versus-fresh operator replay — package 003

Package 003 corrects the JSON signed-zero parsing defect in package 002 and reruns the CPU-only static topology audit with bitwise float-word comparison. The raw capture encodes two zeros as JSON integer tokens `-0`: environment 0, tetrahedron 0, material-frame quaternion `y`; and environment 0, tetrahedron 585, inverse-rest matrix entry `[2][2]`. Python's default JSON integer parser turns each token into integer `0`, erasing the sign bit. The v3 preparer uses `parse_int` to map only the exact token `-0` to floating `-0.0` before binary32 packing.

The regenerated payload preserves `0x80000000` for both entries. Its checker compares every immutable finite float word bitwise, including signed zero, and compares the integer fields exactly. CPU preflight passed all 3,840 environment/tetrahedron records. This resolves the earlier numeric-zero-only audit in package 002; package 002 remains unchanged and documents the intermediate weaker comparison.

The checker was compiled for arm64 against the source101 headers and provenance libraries with `-Wall -Wextra -Wpedantic -Werror`. The preflight command below verifies source, capture, compiled package, metallib, and payload hashes, then loads the package and compares immutable fields before creating any Metal device or pipeline:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/native-cached-operator-replay-360-003/preflight.py
```

The terminal result is:

```text
PASS source101 replay CPU preflight: sealed hashes and all 3840 immutable tetrahedron float/integer words match bitwise, including signed zero; no Metal device created
```

The earlier package 001 and package 002 inputs are retained as `prepared-v1/` and `prepared-v2/`; their original standalone packages also remain untouched. Failed native attempt 365 remains preserved in the result path listed in the manifest.

This is only exact-input preparation and a CPU static-field audit. The source101 Newton 23 capture is a rejected nonlinear state. Package 003 does not run a native operator replay, establish cached-versus-fresh action agreement, or qualify a physical transaction. The parent owns the serialized GPU queue.
