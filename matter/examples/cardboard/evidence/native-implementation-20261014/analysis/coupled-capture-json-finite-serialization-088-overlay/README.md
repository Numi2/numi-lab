# Strict JSON serialization for coupled Newton capture (088 overlay)

This external overlay is based on the frozen source-087 runtime at:
`/Users/home/cardboard-implementation-20261014/snapshots/source-full-contact-arena-capture-087`.
It changes only `matter/src/runtime.mm` manifest serialization. It does not alter the captured binary payload, solver, acceptance, or contact arenas.

## Evidence and repair

The native 289 capture payload is 10,787,124 bytes and its binary SHA-256 matches the manifest. All 140 array descriptors have valid hashes/extents and cover the payload contiguously. Full deformable-contact capture is complete: capacity 18,336, active prefix 100, valid records 100, no unused valid records. However, strict JSON parsing of the manifest fails because `fgmres_layout.vascular_residual_tolerance` is emitted as bare `inf` at byte offset 2568. This parameter is not applicable to this solve because `vascular_unknown_count` is zero. A lenient parse was used only for forensic array verification; it does not make the original manifest valid JSON.

The patch routes manifest FP32 numbers through a locale-independent token formatter. Finite values emit nine significant digits; NaN and infinities emit `null`. The tolerance field adds the exact FP32 word and a semantic label, so the absent vascular tolerance is explicit without rewriting the raw captured state. `dt_seconds` and all four microstep times also carry raw FP32 word fields. The binary payload and all hashes remain byte-for-byte unchanged by this serializer repair.

A regex audit of the 289 manifest, with quoted strings excluded, found one non-JSON nonfinite numeric token: the bare `inf` described above. Other numeric metadata fields in the manifest are integer counts/identifiers or finite floats. The serializer also applies to all floating fields in case a later capture contains a nonfinite timestep component.

## Checks

- The overlay patch applies only to the exact source-087 runtime SHA in `overlay.json`.
- Objective-C++ syntax compilation passed with the source-087 `runtime.mm` compile command and source tree include path.
- Native strict parsing still needs a fresh source-088 capture. Preserve 289 unchanged as the pre-repair invalid-JSON attempt.

To test compatibility against the base snapshot without modifying it, copy the snapshot to a scratch tree and run:

```sh
patch --dry-run -p1 -d <scratch-source-087> < runtime.patch
```
