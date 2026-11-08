# Paired-position standalone rod research: retained result

Status: **research-only, not a production repair**. Main source revision was
`6e9da47e40959a8c8f1a9ca7a0ee342274d561e2`. Apple M6, macOS 27.0
(26A425). Source, generated candidate, binaries, plan and run hashes were
recorded before the corresponding trials. Previous finite-lattice proof,
failed solvers and results remain unchanged.

The temporary native standalone rod entrypoint carries split high/low FP32
positions, hard-anchor targets, axial parameters and timestep to a modified
Metal rod step, and reconstructs double positions on return. For a strictly
straight positive-x, gravity-free, untwisted, hard-anchored, no-contact case,
a paired arithmetic tridiagonal implicit solve checks its local axial momentum
before success. Ineligible cases use the prior research guarded path. This is
an isolated executable and library, not the production code or ABI. The host
candidate has not implemented persistent paired state in MetalWorld, memory
budget accounting, a general 3D solver, or general coupling.

## Replication and holdout

The independently assembled dense FP64 equations used the same inputs as each
run; reference free-row residual and global impulse balance were checked. The
local momentum check comes from central finite differences of the full rod
energy in the native probe at two perturbations, with a 1e-4 m/s agreement
gate. Run each case twice. Detailed per-case gate results and native output are
in `analysis-replication.json`, `replication.jsonl`, `analysis-holdout.json`
and `holdout-corrected.jsonl`.

| Case set | Accepted | All prescribed gates | Identical repeats | Max local momentum residual |
| --- | ---: | ---: | ---: | ---: |
| 5/8 nodes, x axis, offsets 0/0.25 m, three anchor arrangements | 12/12 | 12/12 | 12/12 | 1.241e-7 m/s |
| 6 nodes, x axis, offsets 0.4/1.25/100 m, three anchor arrangements | 9/9 | 9/9 | 9/9 | 1.667e-6 m/s |
| 5/8 nodes, oblique controls on guarded fallback | 12/12 | 0/12 | 12/12 | 0.1565 m/s |

The prescribed maximum local residual is 0.001 m/s. Across the supported 21
cases, maxima were 2.938e-13 m position error, 7.162e-9 m/s velocity error,
1.060e-8 N reaction-force error, 3.777e-13 J work error, and 3.336e-7 m/s
between finite-difference gradients. Acceptance gates and all run conditions
are in `PLAN.md` and `PLAN-holdout.md`. The oblique controls are not routed to
the x-only candidate; their failure confirms the old research fallback remains
unsuitable for promotion. The old and new fallback outputs are not bitwise
identical: the candidate metallib was compiled with `-fno-fast-math` for
compensated arithmetic, unlike the previous guarded build. Both accepted all
12 oblique controls and failed the independent local momentum gate.

The initial 18 six-node holdout invocations all exited 2 before simulation:
the pre-existing harness admitted only five or eight nodes. They are retained
in `holdout.jsonl`. The recorded holdout-plan addendum changed only that probe
input restriction; the Metal candidate was byte-identical, then all 18 valid
runs were recorded in `holdout-corrected.jsonl`. No initial failed launch is
counted as a solver rejection or passing physical trial.

## Scope and next executable work

This demonstrates a promising one-step arithmetic/state representation for a
synthetic straight chain, not a generally qualified rod solver, surgical
simulation, multiple-step stability, complete rollback, supported ownership or
performance improvement. The standalone executable links an instrumented host
entrypoint against the existing native library; the metallib links the newly
compiled research rod AIR with the other pre-existing reconciliation build AIR
files (exact link command in `link-build.json`). The source and binaries are
identified by hashes, but the external build artifacts are not included in
this repository. Non-x and coupled geometries are unqualified and the accepted
oblique fallback is a known correctness failure. Keep this option out of
production. A next candidate must give paired state an authoritative persistent
owner through successive steps and rollback, then address full vector
mechanics with independently checked forces and noncontact controls before any
coupling or promotion.
