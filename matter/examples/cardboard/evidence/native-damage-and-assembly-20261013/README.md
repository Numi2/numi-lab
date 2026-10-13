# Curated cardboard native evidence record

This is the generated evidence record for retained cases in
`/Users/home/cardboard-followup-20261013`. It preserves successful, failed,
incomplete, superseded, and invalid-invocation records. Manufactured fixtures,
literature-grade coupons, and numerical parity checks do not establish physical
cardboard qualification.

To regenerate this record with the shipped packer, run from the bundle directory:

```sh
python3 supplemental/evidence-packaging/package_evidence.py \
  --source-root /Users/home/cardboard-followup-20261013 \
  --output /path/to/new-bundle
```

The retained case archives and their source inputs are read-only. The output
directory must be writable, must be outside the source archive, and is replaced
as one generated directory when the packer runs. The packer does not build or
execute native simulations.

The bundle contains per-case predictions, receipts, results, full CSV/TSV
histories, logs, failure text, plots and JSON manifests. It deliberately excludes
OBJ meshes, compiled `.nmatterpack` archives, executables and Metal libraries.
The aggregate seeded-sensitivity result, source-bound summary/frontier, latest
endpoint-scope documentation delta, Robertsson source-normalization patch,
frontier generator, post-run host observation, and selected build/CPU-regression
logs are retained under `supplemental/` for review. The host observation is
context only; it was captured after runs and is not retroactively part of their
prediction seals. The material-document patch records
two source-equivalence limits: the current fixtures use default area
ratios of 1 rather than per-element projected areas, and the authored ZD
compression energy is shifted to `Psi(I)=0`, so exact energy-level reproduction
is not claimed.
`input-binding-manifest.json` records each retained input path and SHA-256.
Verified UTF-8 source/parameter inputs are stored once each as deterministic
gzip blobs named by their raw SHA-256. Runtime binaries and Metal images remain
hash-recorded with `included: false` and `rebuild_required: true`; they must be
rebuilt on a compatible checkout/toolchain to repeat native runs. Compiler
system headers and OS/driver versions were not archived in the original run
seals.

The packer includes these evidence families:

- `source-point-004` is retained as an invalid instrumentation attempt: its
  fixture perturbed F13 but asserted the tensile ZD F33 tangent. It exited
  before a damage-path experiment, so it is not a material-law failure.
- Robertsson single-ply softwood point tests 003, 005, 006 and seeded
  continuations with one, two and four substeps. The older 003 rejection is
  retained; 005 completes a separate `0 -> 0.3 -> 0` material-point path;
  006 rejects `gamma=0.595` from the accepted `0.59` history. The one-substep
  seeded continuation reproduces that rejected candidate bit-for-bit. Two and
  four substeps accept through `0.595` and return only to `0.59`; their small
  nonzero differences do not establish convergence, and they do not complete
  a high-shear unload-to-zero cycle. These are material-point results, not
  corrugated-board validation.
- `fem-coupling-001`, a fully prescribed one-tetrahedron FEM/point integration
  parity fixture. It has no free mesh degrees of freedom and is not an
  independent material implementation, experiment, or physical validation.
- `cohesive/native-003`, a positive pure-Mode-I paired-face coupon for the
  separate single-ply bleached clay-coated paperboard grade. It does not
  calibrate the corrugated starch bond or mixed-mode/closure behavior.
- Native regression sets 004 through 013. The 008 capsule-box, grip and mutable
  snapshot siblings remain software-regression records, but their frozen-027
  stationary-endpoint observation is not moved-body endpoint proof because BOX
  selectors bypass that gate. Four other 008 entries omitted material arguments
  and are `UNEXECUTED_INVALID_INVOCATION`, not failed mechanics. All five
  corrected 009 invocations exited zero with 360/360 sealed inputs verified
  each. The capsule check exercises moved-body endpoint and rollback validation;
  the cohesive check passed 4,004 native checks for the separate bleached
  clay-coated paperboard grade. The 009 records supersede the old endpoint-guard
  claim without erasing 008. These are native checks, not physical board tests.
- Native-test 010 cohesive reset is retained as an incomplete fixture run: its
  earlier damage/reset/isolation assertions execute, then the final rejected-
  reset target fails runtime admission because the borrowed FEM kinematic-target
  count covers nodes but not every environment-by-node entry. That is a fixture
  setup defect, not a cohesive-law failure. Corrected native-test 011 passes
  4,072 paired-cohesive checks, including history reset/isolation and rejected-
  reset checkpoint verification, with the bleached-board grade limit explicit.
  Its capsule/box sibling also passes Armijo, endpoint, rollback, stationary-box
  and temporary-grip checks. Native-test 012 repeats both fixtures: cohesive
  reset passes 4,072 checks, and capsule/box passes its endpoint, rollback and
  Armijo assertions. The separate scope-034 admission fixture accepts one
  eligible finite-positive-mass step. Native-test 013 passes the cohesive
  reset/coupon checks again (4,072 assertions) and capsule/BOX endpoint,
  rollback, grip and Armijo checks under runtime-policy revision 2. Frozen-036
  scope admission accepts one finite-positive-mass step. The build changes the
  host runtime policy revision; the production Metal library is unchanged.
  These remain software regressions only.
- `box/native-002`, which stopped after 15 accepted native steps. Its fixture
  was externally held and remains unsealed; the manufacturer's adhesive joint
  is not calibrated, and fixture-free physical assembly is false.
- `box/native-003` assembled the full blank mesh (4,308 nodes, 10,356 tets) but
  rejected native step 32 during score-die contact approach after 32 accepted
  steps: status 10, 512 FGMRES iterations, residual 1.24495e-6 against the
  1e-6 tolerance. Frozen-027 `box/native-004` naturally exited 2 after 502.526 s,
  with 32 accepted steps then step-32 contact-approach rejection (status 10,
  512 FGMRES, residual 1.63902e-5). Later folds and closure were not attempted.
  A BOX selector-gate bypass makes the 004 stationary-endpoint path
  `SCOPE_NOT_QUALIFIED`, regardless of accepted steps. Repaired frozen-028
  `box/native-005` used default self-contact and FGMRES 1024, accepted 32 steps,
  then exited 2 after 742.222 s on native step 32 (status 10, residual
  1.25349e-6). No fold or closure actions completed. These are native
  diagnostics, not physical assembly qualification; no calibrated bond is
  inferred.
- `scoring/armijo-001`, a 50 μm native punch-and-backing trial on frozen build
  027. Peak accepted punch force was 0.389654 N at step 47, with two punch
  contacts; 0.388607 N is the later hold-step force. These loading values and
  the first-withdrawal status-10 solver failure after 60 accepted steps remain
  diagnostic. Its stationary-tool endpoint result is `SCOPE_NOT_QUALIFIED`
  because this build's BOX selectors bypass the stationary-endpoint gate. Gap
  and zero-pair fields on the rejected row describe the rolled-back accepted
  snapshot, not the proposed withdrawal endpoint. The run does not establish a
  complete crease cycle or calibrated crease law; the stationary-endpoint
  result remains unqualified because frozen-027 BOX selectors bypass that gate.
- `scoring/armijo-002` and `scoring/armijo-003` are repaired frozen-028
  prescribed punch paths with identical sealed input hashes and compiled
  fingerprint. Their accepted indented-arm CSV rows 0–59 are byte-identical.
  The 24-step withdrawal in 002 rejects step 60 after 60/96 accepted steps
  (status 10, 1024 FGMRES, residual metric 6.80620e-6 to 6.87008e-6). The
  longer 96-step withdrawal in 003 completes 168/168 steps: 48 loading, 12
  hold, 96 unloading and 12 relaxation. It reaches 50 μm commanded travel,
  records peak punch force 0.389654 N at step 47 with two contacts, and ends
  with zero punch contacts. This changes the prescribed schedule at fixed dt;
  it is not temporal convergence or physical crease qualification. Self-contact
  is disabled and the material law is not calibrated to a physical crease.
  Accepted-step certificate residual peaks at 9.99735e-7 on step 30; the
  step-60 status diagnostic near 0.9964 is a deformation/J diagnostic, not a
  residual.
- Solver Armijo-001/004, the Krylov prefix run, and mesh studies, including
  failed and partial runs. The
  mesh-armijo-001 runs rejected, mesh-armijo-002 rejected at the first step,
  and mesh-armijo-003 is an incomplete captured run. They are not admitted as
  mesh convergence. Armijo-004 covers only one fine step; the positive-noop
  case is retained as a separate solver control. Mesh-armijo-004 retained
  nx8 failure after 43/96 steps, nx16 failure after 19/96, and nx32 timeout at
  1200.027 s after 17/96 rows with no native rejection row. These are path
  coverage records, not mesh convergence. `native-tests-009` and later
  corrected siblings are included automatically when their case directories
  exist.
- Frozen-031 `operator-fd-baseline-001` and `operator-fd-001` preserve the same
  naturally rejected no-contact nx16 native path, with only a diagnostic
  selector enabled in the latter. The post-run comparison verifies 360/360
  inputs for each case, identical input maps, CSV bytes, result JSON, compiled
  pack and all 41 exported mesh hashes. At root19/Newton3, the two nonzero
  correction perturbations are admissible (stage status 0, positive minimum
  J), but matrix-free operator action disagrees with centered residual FD:
  relative errors 1.0224 and 1.0248, with action/FD cosines -0.0671 and 0.00175.
  The second environment has a zero correction direction, so its zero error is
  a vacuous control. Both native runs still reject; this diagnoses an operator
  mismatch without changing or repairing the path, and it is not convergence or
  physical evidence. The comparison does not identify the cause; projection,
  active-set behavior, and FP32 effects remain possible contributors. The
  comparison JSON is derived after execution rather than part of either
  prediction seal.
- `operator-fd/accepted-history-yield-audit.json` recomputes Hill `q` from the
  hash-bound accepted material history at step 18, the last accepted row before
  the rejected native step 19. Some accepted tetrahedra are near `q=1`, which
  makes a yield-transition crossing a testable numerical hypothesis. The audit
  does not contain the Newton-3 candidate or either projected `+/-` state and
  therefore does not show that any compared state crossed yield or explain the
  global operator discrepancy. Its cutoffs are descriptive, not constitutive
  active-set flags.
- Host epsilon-scale 8 and 16 captures (`operator-fd-002/003`) bind 361
  verified inputs each; baseline and scale-1 bind 360 each. The added input is
  the diagnostic epsilon-scale option. Both scaled captures preserve the same
  native output bytes and production Metal hash as baseline, and their common
  epsilon 0.25 result repeats exactly. The operator mismatch persists and its
  cause is unidentified.
- `solver/local-projection-accuracy-035` is an isolated alternate Metal build
  using a `2e-6` local projection threshold while production remains at
  `2e-5`. It exited 2 after two accepted steps; the next attempt reported
  status 5 at element 1430 before a microstep completed or FGMRES ran. The
  failure subtype and cause are not inferred, and the tighter threshold is not
  adopted. Its build scripts, metadata, patch, review note and log are pre-run
  context alongside the sealed case inputs, not retroactive parts of the run
  seal.

`case-index.json` preserves each case's status independently. A failed or
incomplete case is not removed because a later run superseded it. The README
and generated manifests state evidence limits explicitly: the bundle contains
no calibrated corrugated kraft/NSSC damage law, no starch cohesive calibration,
and no physical full-box qualification.
