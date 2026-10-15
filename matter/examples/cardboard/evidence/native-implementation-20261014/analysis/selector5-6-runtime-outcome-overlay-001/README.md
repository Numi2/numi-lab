# Selector 5/6 runtime outcome diagnostic overlay

Base: frozen source 129 at `/Users/home/cardboard-implementation-20261014/snapshots/source-hybrid-status-includes-129`; frozen manifest SHA-256 is recorded in `manifest.json`. This package changes only `matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm`.

The source-129 selector parity check threw on selector 5's accepted-status/certificate condition before running selector 6 and did not print the failed selector snapshot. The overlay keeps the same physical fixture, force, selectors, runtime calls, acceptance criteria, and thresholds. It captures exceptions independently for selector 5 and selector 6, runs both runtime instances, prints each outcome before applying the same status/certificate gate, and then preserves the existing nonzero-correction, fixed-node, and representable-motion requirements.

For each selector it prints the runtime status and certificate fields, exact native byte encodings for the complete status vector, complete certificate vector, and FEM node-state vector, plus source/device fingerprints, transaction cursor, and any initialization/encoding/command error. The output is sufficient to distinguish a fixture/setup issue from a selector-specific runtime result without weakening the check. It does not change Runtime or any Metal source.

Validation is CPU-only: the source contract test, Objective-C++ syntax check with `-Werror`, and patch dry-run passed. No build target or native GPU run was performed. Root should integrate this one-file patch into a fresh source snapshot and run the native checker to obtain both outcomes; source 129 and run 420 remain unchanged.
