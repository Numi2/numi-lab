# Native539 command package: source203 all-environment/all-Newton fallback

This is a prepared fresh-from-zero 60-control full-coupon command. It uses
source203 and build-source203-fallback-001. Relative to native535, it changes
the source snapshot/build paths, fresh output path, and
NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR from all:3 to all:all. Mesh, materials
(content hashes), timestep, line search, FGMRES/CGLS budgets, residual
tolerance, all other environment variables, and the physical control schedule
are held fixed. Diagnostic capture selectors remain off.

The command is deliberately gated. runner.py --execute refuses to launch
unless the separate readiness file named by command-spec.json binds reviewed
537 and 538 evidence files with full SHA-256 values and an explicit
approve_for_539 decision. No readiness approval is included here. Native535
ended at a 17-control accepted prefix; it is not a completion baseline. The
source198 native521 output is used only for descriptive prefix comparison.

No full-coupon or convergence result is claimed until a run and its direct
numeric audit complete. Native534 is included only as historical source201
synthetic helper evidence, not as a source203 result.
