# All-trial Armijo summary extent correction (overlay 002)

This is a diagnostic-only follow-up to summary overlay 001. It changes only the preflight extent validation and checked byte accounting in `matter/src/runtime.mm`. The capture stages, arrays, command order, residuals, acceptance, tolerances, and production state are unchanged.

Source119 run393 preserved an early capture skip: the summary expected logical per-environment lengths, while runtime scratch buffers use `checkedBytes()`, which allocates at least one complete typed element and 16 bytes. For the one-environment fixture, `environmentLineSearch` has 4 logical bytes but 16 allocated bytes; the readback copies the entire buffer. `femLineSearch` is allocated per `environment*objectCount`, not per environment alone. The revised admission checks exact scratch extents from those dimensions and includes the full copied source lengths in the preflight. It does not shorten or reinterpret any payload array.

For E environments, O objects, B trial slots, float4 size 16, status size 48, and A=`max(E*4,16)` bytes for the typed environment-alpha buffer, the exact append-descriptor sum is:

- fixed arrays: `160E + 16EO + A` bytes (three status records per environment, final metrics, final FEM line-search, and final environment-alpha buffer);
- per trial: `480E + 16EO + A` bytes (eight stage statuses plus one postdecision status, three metrics vectors, the FEM line-search vector per object, and environment-alpha buffer).

For O=1, these reduce to `176E + A` fixed bytes and `496E + A` per trial. At E=1, O=1, B=24, the capture is 12,480 bytes. At E=4, O=1, B=32, it is 64,720 bytes and 168 descriptors. The existing 32 MiB and 192-array caps remain unchanged.

Earlier 001/002 provisional formulas that treated alpha as logical bytes, treated FEM line-search as one vector per environment without checking object count, or double-counted a status are superseded. `tests/test_descriptor_sum.py` independently enumerates the actual descriptor families and the allocator's typed sentinel semantics. It exercises E=1/2/4, O=1/2/4, and B=1/16/24/32.

The patch applies to the exact overlay-001 runtime hash in `manifest.json`, which was based on frozen source115. Validation is limited to a source-bound descriptor-ledger test and patch dry-run; no Objective-C++/Metal build or native run was performed for overlay002. Native393 remains a retained failed capture attempt with `summary_source_extent_does_not_match_environment_layout`; it is not a payload and must not be admitted by the analyzer. A fresh overlay-integrated capture is needed to verify this correction.
