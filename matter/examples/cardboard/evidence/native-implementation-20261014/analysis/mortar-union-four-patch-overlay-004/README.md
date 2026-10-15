# Exhaustive private self-sweep diagnostics (overlay 004)

This checker-only patch is applied after the sealed whole-sweep mortar patch 003. It adds `--private-self-sweep-diagnostic`, which runs each of the independent private deformable-sweep kernel cases even when an earlier expected gate fails. It writes one JSONL case record per execution, then a summary. A mismatch or per-case execution failure remains visible and makes the process exit nonzero after all cases have run.

The existing `--private-self-sweep` path remains strict and fail-fast. In particular, the three-patch matched-closure positive case still expects `NM_STATUS_SUCCESS`; this patch does not weaken that gate. The diagnostic suite includes that case, four-patch closure, uncovered and mismatched-patch crossings, overlap rejection, sliding, separation, and an exact common-translation positive. The common-translation fixture moves every vertex of both parent triangles by the same FP32 displacement over the kernel's linear endpoint sweep.

Each case emits `numi.matter.private_self_sweep_case.v1` with the case label, expected code and outcome, all returned `NMMatterStatusGPU` fields, and—once fixture construction completes—the exact bytes for every input buffer bound to the production validator. Per-case execution errors retain the constructed input buffers when available. Non-finite diagnostic values are encoded as JSON strings (`"nan"`, `"+inf"`, `"-inf"`) so records remain valid JSON. Each input buffer is separately named and hex encoded in little-endian order, so the record retains the complete kernel input rather than a summary of the geometry. The final JSONL record reports totals and the executable fails if any expected case result differs. Direct kernel execution is covered; this fixture does not exercise a complete runtime transaction.

The exact linear-rotation / later-inward-crossing counterexample remains the analytic CPU case from overlay 003. I did not add it as a native fixture here: the current helper constructs two full triangle primitives and mortar maps, and an equivalent nondegenerate paired-boundary geometry needs a separate authored fixture. The existing native tangential-slide and uncovered-crossing negatives still run in the exhaustive mode. This package therefore does not claim native execution of the rotation counterexample.

## Apply and build

Apply `private-self-sweep-diagnostic-004.patch` to a source tree that already has overlay 003 applied. The patch changes only `matter/tools/cardboard_full_contact_armijo_check.mm`.

The native diagnostic command is:

```sh
cardboard-full-contact-armijo-check <bulk-material.nmatter> --private-self-sweep-diagnostic
```

Capture stdout and stderr together to retain the JSONL records and final exit status. A nonzero exit after all case records means one or more expected status gates failed; it is not a reason to discard earlier or later records.

The host target compiled successfully from the overlay-003 source plus this patch. Metal shaders were rebuilt as a dependency. No GPU test was run for overlay 004; root owns native execution.

## Provenance

`manifest.json` binds this patch to overlay 003's manifest and exact checker bytes, records the patched checker and build hashes, and preserves the compile-only status. It does not revise or replace overlay 003 evidence.
