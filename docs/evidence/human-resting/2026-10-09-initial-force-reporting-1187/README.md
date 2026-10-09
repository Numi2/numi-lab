# Persistent initial-force serializer reporting fix

This narrow change fixes the resting viewer path so it emits the existing initialized persistent-force reference before returning. The non-resting path uses the same serializer. The serializer and field schema are unchanged; no force math, physics, runtime library, or shader changed. The applied source file SHA-256 matches the already-built and smoke-tested source exactly.

The retained native evidence covers 8- and 16-step smoke runs. Each emitted one 128-row force-reference record; the reference was identical between runs and its rows recompute with zero numeric discrepancy. The first two resting-coupled cadence rows match the 1178 baseline exactly, and the corrected 16-step run's initial MRV pack is byte-identical to the baseline pack. See [smoke-validation.json](smoke-validation.json) and [diagnostic-review-v2.json](diagnostic-review-v2.json) for pins and detailed values.

The historical JSON fields retain the `_force_n` suffix, but generalized-force rows mix units: translational coordinates use N and rotational coordinates use N m (N·m). Do not compare all residual magnitudes as one homogeneous unit. This is an initialized force split, not a time-resolved force trajectory or evidence that activation caused later settling.

The earlier 16-step cadence attempt that failed when the viewer rejected step 8 is preserved in the [failed cadence log](failed-cadence-16-step/native.log) with its invocation; its failure is a recording-cadence issue, not a serializer pass. No long run was started.

[external-artifacts.json](external-artifacts.json) pins and rehashes each referenced source/output artifact. [SHA256SUMS.txt](SHA256SUMS.txt) covers every file published in this directory. No MRV packs, binaries, model assets, or build outputs are duplicated here. The existing build and short-run evidence are referenced by hash; this publication checkout was not rebuilt.
