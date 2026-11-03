# GPU-active compaction and whole-skin/fat conditioning witnesses

All builds and execution: SSH Numi, Apple M6, macOS 27. Authoritative source/binary pins, launch plans, clocks, compact traces, resource receipts and full-stdout hash receipts are retained. No visual verification was performed.

## Active-work fix

The tiled compaction introduced in 992738b4a retained capacity-sized launches for empty or small candidate lists. Generate the single-environment tile dispatch directly from min(GPU candidate count, capacity), using the existing indirect-dispatch buffer after narrowphase and before the active-contact scan overwrites it. The tile prefix reads only active tile totals, so stale/poisoned inactive scratch is never consumed. Multi-environment processing keeps its deterministic padded layout. Contact identity/order/history and forces are unchanged. Kernels require their checked 256-thread tile width; unsupported widths fail closed.

The complete checker passes **790,741 checks**, including exact serial/reference compaction parity for single/multiple environments, GPU dispatch dimensions, empty/partial/overflow-clamped counts, poisoned padding, canaries and untouched tails. The prior thin-contact law and residual/certificate regressions also pass.

Three matched, alternating-order GPU benchmark pairs include all four passes and barriers. Empty work falls from the prior capacity-sized median 1.049 ms to **0.00642 ms**. Legacy single-SIMD empty work is still faster at 0.00282 ms; the small constant multi-pass overhead is retained honestly. At 3,631,873 candidates, serial median is 31.343 ms and tiled median 3.856 ms. These are compaction-stage results, not whole-scene speedups.

## Native comparison

The final native build completes two requested steps at 15.625 microseconds. Compared with the retained completed-zero-action two-step run:

- Every coupled CSV field is identical.
- All **318,974 endpoint node rows** are identical, including high/low displacement, velocity, mass and force/residual channels.
- All **1,521,713 endpoint tetrahedron rows** are identical.
- Accepted body-state, respiration-state and full rendered vertex-buffer hashes are identical.

Complete endpoint stdout is approximately 801 MiB per successful trial and remains source-pinned on SSH. The compact publication omits only node/tet rows and retains their complete ordered hashes plus full-file hashes.

The application reports **6.9645e-7x**, including its 44.87-second simulation/export segment. The first two accepted GPU command durations are 18,202.30 and 18,751.59 ms. This is a very short run, with initialization excluded from that application factor. It is not supported rest, motion qualification, sustained throughput or 0.1x.

## Bounded geometric search

The resting native source uses the existing maximum of 32 backtracking candidates instead of 12. Existing parallel pending-object selection avoids running later candidates for objects already accepted. No determinant/material/physical acceptance bound changes.

The retained 16-step bounded-backtracking/reuse trial keeps the same nine accepted steps as before, passes the previous skin upper-J search stop, then fails the nonlinear certificate at root 9. Its nonlinear norm plateaus around 4.5754e-4 across later iterations; no step is accepted from that plateau. Do not raise Newton budgets further on this evidence.

Existing exact-zero contact geometry reuse and completed-zero operator suppression are enabled only through their original admission checks. The separate completed-zero-action evidence directory contains its matched-prefix result.

## Whole-body mechanical conditioning screen

An independent reference-cell calculation over all 1,495,294 imported skin/fat cells finds:

- Minimum tetrahedron altitude: **1.584 nm**.
- 1,107 cells below 10 nm; 17,654 below 100 nm.
- 28,168 edge matrices have 2-norm condition number above 10,000.
- 32,500 cells meet either the 100-nm altitude or 10,000-condition diagnostic screen.

These are diagnostic thresholds, not new anatomical validity limits. Positive finite cell volumes and zero surface crossings do not establish a well-conditioned deformable discretization. No cell was deleted or mass redistributed by this audit.

The exact root-9 upper-J witness, global tet 1456796, has a 4.732-nm minimum altitude, 14.8-58.1-micrometre edges and condition number 23,806 in the emitted Float32 reference. This is concrete evidence of a difficult local approximation; it does not alone prove the complete convergence cause. Its four accepted step-2 node rows and region/source-face ancestry are retained.

A tighter-existing-linear-forcing experiment is running from the same scene to distinguish linear direction error from an infeasible/poor mechanical approximation. Its result is not claimed here. The remaining solver/model repair and >=0.1x target are open.
