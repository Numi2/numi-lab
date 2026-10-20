# GPU timing profile 031 interpretation

The contact-selected capture gives a clear ranking only within the sampled root-1 window. Each selected kernel has 100 elapsed-time samples, all on step 1:

| Contact kernel | Samples | Sum of sampled GPU intervals | Median | P95 | Share of selected-kernel sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nm_contact_sort_surface_primitives` | 100 | 1.003474 s | 10.035 ms | 10.038 ms | 73.4% |
| `nm_contact_compact_deformable` | 100 | 0.300478 s | 3.004 ms | 3.006 ms | 22.0% |
| `nm_contact_scan_deformable_node_incidence` | 100 | 0.063582 s | 0.632 ms | 0.632 ms | 4.6% |

Together these measured intervals sum to 1.367533 s for those three kernels in the captured root. This suggests surface-primitive sorting is the largest of these selected costs in this short window. The tightly clustered per-kernel samples show low variation within this one root, but there is no multi-root distribution here.

The sparse profile sampled all selected kernels at a 256-dispatch interval and retained 630 GPU timing samples across 37 stages, all on step 1. It reports 224 dispatch-count records (339,056 summed reported counts across steps 0 and 1). For contact, it captured only five `nm_contact_build_surface_bvh_level` samples (12.999 microseconds summed); it provides no useful sample set for ranking the three kernels above. Sparse sampling therefore does not support a contact-kernel ranking in this run.

The broad Matter-stage timestamps are not stable across these two instrumentation configurations: `matter_after_microstep` sums to 468.160 ms over four sparse-profile samples, versus 7.393 ms over four selected-contact samples. `matter_after_frame_begin` is 209.491 ms in the sparse profile and 3.393 ms in the selected-contact profile. These are GPU command-buffer timestamp intervals, not CPU durations, and their large configuration dependence makes them unsuitable as a whole-root GPU cost estimate.

Both commands exited 0 after two accepted 125-microsecond rows. Their process wall times were 27.911 s (sparse) and 27.831 s (contact), which include launch/setup and do not isolate CPU time. Neither package records CPU utilization or per-stage host timing, so a whole-root CPU/GPU split cannot be calculated. The profiles cover root 1 only; they do not characterize a longer run. No explicit lost-sample marker was emitted, but absence of that marker does not establish that every eligible sample was resolved.

A fixed allowlist of 30 completed `resting-human.mrvpack` outputs was checked by SHA-256 and size (297,078,784 bytes each). It contained three exact content groups of 10, 3, and 17 paths. Thirteen singleton paths in the 17-path group were atomically hardlinked to its existing canonical file; all 30 original pathnames remain, and the 13 replaced targets were verified to share the canonical inode. The nominal logical bytes unlinked were 3,862,024,192 (3.60 GiB). Available space rose by 3,771,672 KiB between the final preflight and post-transaction `df -k` snapshots, within about 164 KiB of the nominal amount; a separate smoke run was active concurrently, so the observed filesystem delta is not attributed solely to deduplication. The exact allowlist, per-path before/after inodes and hashes, and df snapshots are recorded in [storage-dedup-033.json](../../storage-dedup-033.json). The concurrently active `native-vascular-block-smoke-033` path was excluded.
