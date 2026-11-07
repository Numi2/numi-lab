# Native resource kernel compatibility and runtime recovery

Kernel ABI4 uses `mr_finite_resource_commit_v4` and a matching dispatch guard.
State/program ABI3 and its existing80-byte state/fingerprint semantics remain.
Two stale shaders reject with the new library before resource encoding; the
previous library rejects the new shader. Four crow executables now embed only
the relative adjacent-library path. A missing adjacent library fails before main.
Actual dyld library paths/hashes are checked rather than inferred from a command.

The matched runtime reproduces all28 numerical reference files exactly over the
existing nine448-tick physical trials across eight lanes and synthetic controls.
Summary fields match except elapsed time. No physics tolerance or task changed.

Two invalid instruments are retained. The initial copied executable used an
absolute build-library path and bypassed the intended stale shader; it completed
a normal current-runtime admission instead. A second instrument passed all three
shader mismatch arms but reused a dyld launch closure across executable hardlinks
in its missing-library arm. Distinct executable inodes correct this isolation
issue; immutable library/shader payloads may share local storage during captures.

The earlier spatial Git recovery verified stored contents but did not prove which
library the native replay loaded. A supplementary fresh process restores the
three historical native roles directly from Git, explicitly selects its restored
library and checks actual loader identity. All28 numerical references match.
This repairs that native spatial replay qualification; other older native
recovery instruments have not been newly audited here.

`manifest.json` retains every selected raw file and runtime variant through
content-deduplicated Git objects. `runtime-recovery.json` selects the current
eight native roles, exact learner source blobs and29 reference files.
`restore_runtime.py NEW_DIRECTORY` restores only those selected files from Git.
`verify_recovery.py` replays the current runtime, checks its actual loaded library,
compares the reference files and removes verified temporary duplicates. The same
hardware, OS and installed dependencies remain required. Source/runtime identities,
plans, original loader logs, failures and scripts are retained. Cleanup receipts
bind exact paths/hashes and open-handle checks; bulk checkout duplicates are omitted.

This is engineering compatibility and recovery admission. No new behavioral,
biological or performance qualification is claimed. The next behavioral work is
two reachable food sites and a causal visible/occluded cue in the native renderer.
