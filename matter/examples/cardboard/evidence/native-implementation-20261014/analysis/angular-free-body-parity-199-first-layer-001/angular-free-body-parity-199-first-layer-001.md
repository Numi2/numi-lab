# Native 199 first-layer diagnosis (source 064)

## Evidence binding

The retained failed attempt is `/Users/home/cardboard-implementation-20261014/native/angular-path-free-body-parity-199/`. Its prediction records the frozen source-064 snapshot, the exact full-contact checker and `--sparse-free-body-parity --extended-budget` argv, `execution_started=true`, unchanged inputs, exit 1, and no timeout. Hashes:

- prediction: `d5fd4dba88cd01b6367bebb4e31dd1b7515ec295bfa4c1de62d653ce51f80e74`
- stdout: `b9cfd2240a76fe8947ed52537aec88565a2cea7e0250189d17e7c00cacd786fa`
- stderr: `59b4a184325f705afbc8da630f1b070a237c1a2051e9e79eef81032177f77aeb`
- receipt: `e897fde0c12face184a67dfaf8265976b8e78fcd618fab77dac012cf5dabc70e`
- source-064 `matter/src/runtime.mm`: `32597ceffdafc88ed507b2e6bc06ff6a78bffc7918a20884f0586387153e3f8e`
- source-064 `matter/src/metal/fem.metalinc`: `24e88be8674ffa2f2c859be29c564cd99c97f8fd5946b16dcea3aebab5e1fee3`
- source-064 `matter/src/metal/contact.metalinc`: `69c3ec930d0b382dfe278fa42c56353782438ded77d8224a3ccfd6989e3067c9`

## Observed failure

Control step 1199 was accepted with nonlinear certificate residual `9.34852622e-8`. Control step 1200 then failed and rolled back exactly: before/after hashes match for FEM nodes, cohesive faces, rigid states, rigid contact history, and deformable contact history. The terminal Matter status is nonlinear solver failure (code 10) at Newton index 11, with base residual `1.42294965e-7`, last tested residual `1.42295093e-7`, last alpha `1.1920929e-7`, and 24 trials. The runner reports its enclosing environment failure as status 11. This is a failed run, not an accepted physical step.

## First concrete measurement boundary

The line-search JSON reports a pre-limiter trial. In source-064 `runtime.mm`, the private residual-Armijo loop runs at lines 15913–16087 and its trace is emitted at 16089. After that, runtime dispatches the MPM line-search selection, synchronizes FEM alpha, runs deformable CCD and rigid CCD limits, and synchronizes alpha again (roughly lines 16120–16300). The final free-rigid and FEM updates consume those post-limit buffers at lines 17099–17120. `nm_fem_synchronize_environment_line_search` reduces object alphas (source-064 `fem.metalinc:4431–4451`); the contact limit kernels each publish `min(existing, bound)` (source-064 `contact.metalinc:6591+` and `7021+`).

Accordingly, the logged accepted Armijo alpha is not guaranteed to be the alpha that produced the next Newton base residual. The first exact-trial/next-base mismatch is Newton 5: trial alpha `0.0625`, trial L2 `1.41007575e-7`, next base L2 `1.41997361e-7` (difference `9.8986e-10`, about `0.70%` of the trial norm). Newtons 0–4 have exact trial-to-next-base norm continuity. Later small mismatches persist. The ordering is a concrete explanation candidate, but the run did not capture the final alpha after each limiter, so it does not prove which limiter changed it or prove a material/history defect.

## Next diagnostic

Rerun the same source-064 attempt with `NM_FEM_NEWTON_TRACE_ROOT=1200` and inspect the existing `before_contact_limits`, `after_deformable_limit`, and `after_rigid_limit` line-search records for Newtons 5–6. Compare these alphas to the preceding Armijo `selected_alpha`. If unchanged, then pursue private-trial versus committed-candidate reassembly, including material and rigid candidate identity. If changed, the accepted-trial metric is simply pre-limit; move the residual descent check after the final admissible swept bound or re-evaluate the residual at the final applied alpha. Keep the CCD bound, material history transaction, and final `1e-7` acceptance gate unchanged.

## Correction (source-064 branch distinction)

The limiter-after-Armijo sequence above applies to the no-capsule/no-box residual-Armijo branch (`runtime.mm:15915` onward). It is not the ordering used by the capsule/box full-contact branch. That branch runs its residual-Armijo loop after the contact limiters and environment synchronization (`runtime.mm:16424` onward); there is no subsequent CCD limiter before the normal free-rigid/FEM apply at `17099` onward. Therefore a post-Armijo CCD limiter is not an explanation for the Newton-5 trial/base mismatch in the full-contact run.

A separate alpha mismatch remains testable: `nm_fem_armijo_decide` writes the FEM object alpha (`state.femLineSearch`), while `nm_rigid_apply_candidate_solution` reads the distinct environment alpha (`state.environmentLineSearch`). The existing trace records the former but not the latter. The source-065 diagnostic overlay at `analysis/free-rigid-armijo-alpha-trace-overlay-001/free-rigid-armijo-alpha-trace.patch` adds a read-only capture of the environment alpha after Armijo, before application. Until a source-bound native capture compares the two values, this is an unresolved mechanism, not a confirmed cause.
