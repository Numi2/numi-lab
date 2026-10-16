# Source249 contact compaction checker-fixture correction

Source249 is a fresh candidate based on immutable source247. It changes one checker-only initialization in matter/tools/cardboard_contact_sample_compaction_check.mm: the ordinary compact FEM residual now starts at the all-zero vector, matching the dense reference. The overflow path retains its exact (17,18,19,0) sentinel. No production kernel, page storage, gather, or admission logic changed.

Native592 failed at the compact/dense FEM residual byte comparison because the normal compact buffer began with (0,1,2,0) per node while the dense buffer began with (0,0,0,0). The receipt confirms the failure occurred after execution began, and source/build inputs were unchanged. This identifies a fixture mismatch; it does not establish a production compaction defect or qualify paging.

A fresh source-bound build of the probe and four page checkers passed, and all five applicable CPU contract checks passed. This task did not execute the native Metal checker. Root can run the compiled checker with:

/Users/n/cardboard-implementation-20261014-host-macmini/analysis/source249-contact-compaction-fixture-001/build-001/numi-matter-cardboard-contact-sample-compaction-check /Users/n/cardboard-implementation-20261014-host-macmini/analysis/source249-contact-compaction-fixture-001/build-001/shaders/cardboard_contact_sample_compaction_check.metallib

The full shipping trajectory and native paging remain unqualified.
