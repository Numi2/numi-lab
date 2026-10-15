# Erratum: selected trace state packing

The earlier selected-point trace packing form was unsafe if its state-copy loops used `NM_MAX_MATERIAL_STATE` (32) instead of the actual 15-state material count. The reconstructed faulty loop is:

```metal
for (uint i = 0u; i < NM_MAX_MATERIAL_STATE; ++i) {
    output[41u + i] = projectionState[i];
    output[56u + i] = projectionLow[i];
}
```

This is a reconstruction of the defect pattern, not a byte-preserved historical source artifact; the earlier faulty file/hash was not present in the sealed 086 overlay. With 32 iterations, the high and low ranges overlap at slots 56–72; high/low writes also collide with the two sentinels at 71–72; low-state writes 73–87 overwrite the first 15 trace slots. The original writer order determines which value survives, so the resulting trace could not be interpreted reliably.

The sealed 086 source uses a 15-iteration loop. It writes high state to slots 41–55, low state to 56–70, sentinels to 71–72, and the 5,992-slot trace to 73–6064. The host allocates 6,065 floats, checks the sentinel words, and serializes that same trace range. The standalone CPU sentinel test in this directory verifies the source hashes and exercises a 32-state backing array while rejecting overlapping destinations.

This erratum and test establish source-level packing consistency only. They do not establish that the native diagnostic ran successfully or that the underlying material failure was reproduced.
