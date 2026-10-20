# Unified verifier 026 — rollback/retry passed

The corrected verifier completed the CPU and geometry/interface suite, then ran a two-step native rollback/retry check. Native execution exited 0 in 28.559 s. The injected downstream rejection was followed by `shared_tissue_late_reject_retry=passed`: the log records unchanged generalized coordinates/velocities, matter/vascular state, passive-adapter history, brain/respiration presentation, restored body/DOF properties, and a same-context retry. Both accepted CSV rows are retained in `native/resting-coupled.csv`.

The two accepted rows show zero normal contact impulse and zero root assistance; this is a short rollback regression, not sustained support or physiological qualification. Terminal FEM audit reports 2 accepted steps, 3 adapter passes, min/max J 0.717165480643 / 1.10540500724, and maximum displacement 0.00057885402099 m across 22 objects (14,302 nodes; 54,565 tetrahedra). All 22 per-object summaries and the terminal global J audit are in `native/selected_stdout.log`.

All CPU/geometry/interface receipts and compact logs are retained. The large run pack, raw full stdout (which contains per-node/per-tet arrays), binaries, and bulk geometry CSVs are omitted. Native input paths and their hash/size list are bound to `native/input-assets-reference.json`; native source and executable hashes are in `native/source-binary-binding.json`; all 11 overlapping source entries match the preparation receipt.

Attempt 025 is preserved separately as a verifier setup failure: the contact-geometry helper path did not resolve, so no native app run occurred. Per the task handoff, 026 corrected that script path while leaving native physics source/shader unchanged; see `attempt-comparison.json`.
