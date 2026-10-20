# Numi Human Metal performance experiments on SSH Numi

This report records the implementation and measurements performed on SSH numi (Apple M6, 12 GPU cores, macOS 27) against baseline source aadcd68e0747540978edd88a23af0cd114932942. All native builds and runs were executed on that host.

## Changes and measured results

The runtime now skips the mixed-field smoothing subgraph only when the compiled scene has no multiphysics objects and no field boundaries. Single-environment FGMRES work uses GPU-authored indirect extents for the FEM-node and tetrahedron domains so converged columns do not execute the large vector operators. A large, single-environment FEM scene with at least 8,192 nodes uses a 512-lane FGMRES orthogonalization reduction; smaller and batched scenes retain the existing 256-lane path. The physical equations, convergence tolerances, accepted-state transaction, and rollback policy were not changed.

The sorted contact-incidence option was measured in two matched pairs and ran about 29% slower, so it remains disabled. The existing tangent-block preconditioner trial was interrupted after more than 143 seconds without an accepted root; it is censored and provides no speed result.

The empty-field skip (067) and GPU-indirect FGMRES (068) each passed four accepted roots and the FEM/fascia audits. Their endpoint metrics matched both accepted baseline controls within the existing 1% comparison. The indirect replay encoded 221,184 node and 49,152 tetrahedron indirect dispatches. Its body real-time factor was 1.0030e-5× and process wall was 61.33 s, a small improvement over the baseline median of 9.502e-6× and 64.04 s.

The 512-lane run (070) accepted all four roots and showed the speed gain. The final width-aware implementation (073) repeated the four-root run on the exact code being published: all roots were accepted, FEM/fascia audits passed, and root assistance remained false. It measured 1.1138e-5× body real-time factor and 56.64 s process wall, about 17.2% more throughput and 11.6% less process wall than the two accepted baseline controls. The endpoint comparison is not a full pass: peak anchor reaction was 2.417694 N versus 2.445168 N (1.12% difference), slightly outside the pre-existing 1% numerical-consistency gate. All other captured endpoint metrics were within that gate. Preserve that miss when interpreting this candidate; the solver's own accepted-state and tissue audits passed. A follow-up 1,024-lane trial (074) completed but was 2.1% slower than 512 lanes and missed the same comparison gate on peak deformation (max_J 1.12935 versus 1.11625, 1.17%). It is rejected; 512 lanes remain selected.

## Profiling and timestep checks

Matched one-root profiling measured 18.607 s unprofiled versus 18.645 s with selected-kernel instrumentation, a 37.5 ms (0.2%) wall difference; endpoint metrics matched. The low-rate selected-kernel capture sampled every 128 calls without counter drops. The all-kernel capture recorded 111 kernel dispatch-count names and 210 timing samples without counter drops. At one root it counted 6,144 invocations each for FGMRES orthogonalization, FEM operator, gather, patch preconditioner, and vascular operator; deformable node-incidence count/scan/scatter each had 100 calls. Sample-scaled timings put orthogonalization first and patch preconditioning second. Those scaled values are diagnostic estimates, not exact total GPU time: many sampled calls are fast converged-column returns, and the sampled work varies substantially.

Two larger-step trials covered the same 0.5 ms simulated interval: 0.25 ms × 2 and 0.5 ms × 1. Both were accepted by the solver and passed tissue audits, but both changed contact, penetration, ligament-force, or body-velocity endpoints beyond 1% versus the 0.125 ms controls. They are not accuracy-qualified, and the configured 0.125 ms step is retained.

A one-root replay without the mechanics-only flag was also captured. It accepted one 0.125 ms body step and passed the FEM/fascia audits; the render path reported 21.44 ms GPU time and presentation 31.76 ms wall time. Its terminal-presentation record reports zero controller steps, physiology is only marked clock-matched, and presentation qualification remains pending. It is not a coupled physiology or interactive-viewer qualification.

## Validation and limits

The native target built with the M6 Metal compiler. The five targeted tests passed: mixed MPM/FEM, monolithic multiphysics, production transaction rollback, FGMRES true-residual kernel contract, and native contact-incidence sort.

The requested 0.1× target is not reached. The final four-root mechanics replay measured 1.1138e-5×, roughly 8,980 times below that rate. At the 0.125 ms root step, the 0.1× budget is 1.25 ms per root, while measured GPU root spans were 6.05–16.00 s. The test horizon was 0.5 ms; there was no five-minute coupled run, no respiratory intervention comparison, and no controller or physiological response qualification. These runs establish software execution, accepted mechanics, and performance measurements only.

Per-run invocation.json files preserve exact arguments, environment, source and binary hashes, input hashes, and timestamps. Compact stdout/stderr traces, summaries, counter samples, and rejected/censored results remain beside this report. Large generated render/export outputs were kept out of this Git increment to avoid publishing redundant multi-gigabyte snapshots.
