# 069h generation provenance

069h is a diagnostic bilateral hand-skin candidate assembled from the audited 069d left payload and saved 069g right-side scale-0.2 state. It is not physically qualified. The right source-angle preference was explicitly waived for this diagnostic: edge (887, 888) improves from 91.1139° to 89.5863° while absolute deviation from the registered source angle increases 27.6426° to 29.1703° (+1.52765°). This is recorded as a waived construction preference, not a passed gate.

## Reproduce from retained assets

On `numi`, run from the evidence root with a fresh output folder name:

```sh
cd /Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001
/Users/n/numi-human-prep-venv-20261005/bin/python finger-profile-069h-right-scale020-preference-waiver/build-069h-reproduce.py finger-profile-069h-reproduced-001
```

This entrypoint reads the retained 069d payload and 069g saved state; it does not rerun the optimizer. The exact executed assembly source is `build-069h-assemble-executed.py` (SHA-256 `a5567d18a4f952f303d95a81b5ea72f09763a6bcb59a717eea82c98e68011f41`), byte-identical to `/tmp/emit_069h.py`. `generation-provenance.json` pins the upstream 069d ancestry fix, corrected 069g baseline constraints, saved fit arrays, source/runtime inputs and 069h outputs by SHA-256. Failed 069e/069f/069g diagnostics remain indexed and unchanged.

The retained eight-pose surface audit reports `passed_source_and_retained_wrist_pose_checks_not_physical_qualification` across 8 poses. Its exact scope and limitations are in `retained-eight-pose-surface-audit-069h.json`; this does not establish physical or layer-mechanics qualification. The 26 MB NHSKIN and 48 MB geometry archive are hash-referenced and not included in code staging.
