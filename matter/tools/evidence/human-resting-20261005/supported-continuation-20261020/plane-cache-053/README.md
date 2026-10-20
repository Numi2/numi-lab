# Supported human contact and bounded solver evidence

The sole deformable-skin bed path now accepts the original supported pose on SSH Numi / Apple M6. This is a short mechanics regression, not completion of whole-body contact or the 0.1× performance target. No visual verification was performed.

## Changes and ownership

All 2,332 admitted outer skin nodes contact the existing flat plane. The bonded three-layer skin/fat shell transfers loads through existing finite-elastic attachments. Articulated bed rows are removed during execution; their static solve supplies only the recruitment initial guess. The compiler's source-only `rigidContactEnabled` admission flag leaves the package/GPU ABI and continuum contact unchanged.

Fixed-plane rounding guards now use the arithmetic forming `dot(normal,position)-offset`. Tangential coordinates no longer inflate the distance guard. The failed 052 trial is retained: its 1.616 µm collision thickness was a numerical floor, while authored clearance was 1 nm. The corrected arithmetic accepts the original 89.66 nm positive gap without translating anatomy. No masses, material laws, acceptance thresholds, assistance forces, or support geometry were changed.

The supported-pose caller enables the tested bound preventing optional joint-stop unloading from worsening recruited balance. Initial normalized residual RMS is 2.23475e-5. This remains an initialization guess: 48 muscles are near maximum activation, and settled physiological equilibrium is unqualified.

Valid FEM tangent-cache applications defer unused base-state geometry; stateless cache columns reuse stress validity from column zero; nontransport field actions skip exact-zero gathers. The block preconditioner respects object scheduling. Its opt-in cached specialization reuses existing scratch and falls back to the general material evaluator on cache misses. Runtime material evaluation interprets compiler-derived programs, rather than performing automatic differentiation. Mode 6 remains optional.

## Measured evidence

| Check or run | Result and scope |
|---|---|
| Six focused compiler/native checks | Passed: rigid admission, fixed-plane arithmetic, support recruitment, registered ligaments, material-frame cache parity, and nonzero coupled fields |
| Scheduled two-object block check | Passed: non-due mass blocks, due stiffness, direct/cache solution parity, bitwise cached-block agreement; tiny load, not tissue validation |
| Scalar control | All 77 trace fields match 052 exactly |
| Eight accepted skin-bed roots | 1 ms simulated; minimum J 0.70617; maximum tissue displacement 2.3623 mm |
| Two-root forced late rejection and retry | All 77 accepted trace fields exactly match ordinary replay and the eight-root prefix |
| Final diagnostic-only rebuild | All 77 fields match the preceding physical run |
| Four Newton / 16 Krylov iterations | Rejected before any accepted root |
| Four Newton / 64 Krylov iterations | Same rejection: residual 4.799e-4, correction 3.040e-2; configured budgets were applied |
| Optional mode-6 trial | Interrupted at 60 wall seconds with zero accepted roots; inconclusive, not a physical rejection |

The instrumented four-Newton / 64-Krylov replay retained 44 existing vascular trace records. On the first microtick, each of four Newton steps accepted alpha=1; reassembled residual decreased from 0.007364 to 0.000925. The inner solve used at most 15 columns there. The final failure receipt exactly matches the uninstrumented run. This identifies insufficient outer convergence under the four-step cap; it does not validate a replacement solver. See `budget-trace-analysis.json`.

The bed impulse uses the existing rollback-owned frame reduction. First-root mean normal reaction is 0.07517 N over 125 µs. This demonstrates initial contact, not settled bodyweight support. Terminal contact history can contain no loaded rows when skin is non-due on the terminal microstep; it is not an integrated force measurement.

The per-object audit covers 14,302 nodes and 54,565 tetrahedra in 22 objects. The aortic wall is the dominant early transient: 2.3623 mm displacement, 1.4486 m/s speed, minimum J 0.70617. The skin/fat shell moves 26.1 µm with minimum J 0.98524. The initial aortic wall has no load-equilibrated prestress state under approximately 85.45 mmHg. Its pressure traction must remain active when repairing initialization. See `terminal-object-audit.json` and `next-blocker-prestress-review.md`; the existing material-parameter staging API does not establish equilibrium.

The eight-root warmed window measures **1.1600e-5× real time**, excluding loading and the first accepted root. The complete physical loop measures 1.1226e-5×. The 0.1× goal remains unmet. Reduced work budgets and interrupted trials are not counted as speedups.

## Coverage and reproduction

The catalog contains 34 registered tension-only ligament bands, including four lumbar interspinous bands retained in `lumbar-interspinous-052`. Regional anchors and effective material parameters are inferred reference choices, not measured entheses. The 26 fascia regions comprise six sourced pectoralis regions and 20 inferred routes; they are not 26 anatomical ligaments. Knee collateral/posterior, wrist/hand, capsule, broader spine and costovertebral coverage remains incomplete.

Run `tools/verify_human_supported_tissue.sh` on SSH Numi. It pins the 34-row actual-pose audit and the same skin-bed template for both vascular policies. Optional 32-step native and two-step rollback modes are short regressions. This session ran focused checks and the recorded native commands; it did not execute the entire wrapper end to end. Invalid-cache-bit corruption is not separately tested.

Per-run invocation, provenance and execution files retain commands, source/binary/asset identities and timing. Frozen patches and compiled-source input records distinguish builds. Failed builds, fixture attempts, physical rejections and interruptions remain available. Compact packages omit large immutable packs while retaining identities; storage receipts record verified hardlink deduplication with all paths and contents preserved.

Full contact/ligament coverage, settled support, long-horizon stability, physiological plausibility and five-minute integrated qualification remain open. No clinical or visual qualification is claimed.
