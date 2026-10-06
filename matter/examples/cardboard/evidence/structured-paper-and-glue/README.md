# Explicit paper, corrugation and finite glue

This second-stage record starts from source commit
`b1cfa538c1b4f9e75710184a32648afa734ebaac`. The native device is Apple M4.
These are numerical development checks, not physical validation. The recipe
combines published values from different specimens, with assumptions retained
in the [manufacturing manifest](../../manufacturing-source-manifest.json).

## Question and instrument

Can an explicit corrugated structure with orthotropic plastic paper and finite
starch-glue regions complete a native loading/unloading cycle, preserve plastic
history, and keep an unloaded reference free of spurious plasticity? This tests
material integration. It does not demonstrate a freely unloaded crease or a
folded box: both grips remain clamped at zero angle during the last phase.

The default coupon has two 0.277 mm liners, a 0.191 mm fluted medium, 7.9 mm
pitch and 4.210 mm board caliper. Upper and lower finite glue bridges have
different widths and gaps. The starting state is preformed, bonded and
conditioned, not the result of simulated roll forming, curing or drying.
The paper law is an associated, compression-calibrated Hill ideal-plastic
model in Green strain. Hardening, asymmetric tension/compression yielding,
paper fracture, interface debonding and moisture evolution are absent.

## Geometry evidence

`finite-glue-compile-001` has 588 nodes and 960 tetrahedra on a coarse mesh.
The independent FP64 exported-geometry audit finds one connected component,
positive cell volumes, at most two incident tetrahedra per face, and only
liner–glue and medium–glue material interfaces. It counts 32 shared triangular
faces of each interface type. Glue volume from the tetrahedra agrees with the
sum of bond cross-section areas times board width within 8.5e-16 m³.
The three-material coupon mass is 0.394251 g. This is a discretized model mass,
not a measurement of the source board.

The separate Nagasawa arc-profile check preserves the published dimension
closure mismatch under its stated centerline-radius interpretation. It does
not silently tune the radius to match caliper. The executable recipe remains
an explicitly declared sinusoidal approximation.

## Constitutive evidence and rejected hypotheses

The production Metal material instrument calls the same projection and
algorithmic tangent functions used by Matter. An independent FP64 material
oracle, compiled-bytecode FP32 trace, and finite differences provide separate
comparisons. None of these material-point tests takes physical timesteps.

| Record | Observation and revision |
|---|---|
| `paper-calibration-001`–`002` | Rejected even at rest. The compiled derivative divides by q²; q-floor 1e-8 put this below the production scalar evaluator's 1e-12 denominator guard. |
| `paper-calibration-003`–`005` | A q-floor of 2e-6 fixed rest, but mixed perturbations at a yielded liner state still failed. Increasing local Newton iterations alone did not fix it. |
| `paper-calibration-006`–`007` | Trial-stress initialization alone also did not fix it. The FP32 trace showed finite equations and nonsingular pivots, but exhausted all eight Armijo backtracks. |
| `paper-fp32-trace-002.log.gz` | The liner needed a first step of 1/256: the ninth trial. The medium needed 1/64. |
| `paper-calibration-008` | Sixteen bounded backtracks passed rest, compression, three shear directions, rotation covariance, stress/strain comparison, tangent differences, and stress-free unloading for both papers. |
| `paper-calibration-009` | Multiplying all implicit residuals by ten tightened local accuracy without changing the constitutive zero set. The largest tested tangent FD relative discrepancy decreased from 1.37% to 0.193%. |

The canonical materials use the final numerical regularizers and scaling.
The material language now allows `update` to initialize a state that also has
an `implicit` residual; that residual remains the acceptance equation.
Duplicate roots are still rejected. Local Newton iterations are explicitly
bounded and fingerprinted. Existing material and stateful regression checks
are retained in `final-cardboard-ctest.log.gz`.

The global board did **not** become successful merely by tightening the local
residual. Its first 100 µs loaded step failed with residual approximately
1.48e-4 (7 global Newton iterations), 1.43e-4 (14 iterations), and 1.41e-4
(tighter material residual), versus the unchanged 1e-4 global tolerance.
Disabling self-contact produced the identical failure. Those trials remain
in `finite-glue-plastic-diagnostic-001`, `finite-glue-plastic-cycle-001`,
`finite-glue-plastic-no-contact-001` and `finite-glue-plastic-scale10-001`.
Thus local accuracy and contact were not established as the main cause of the
global rejection. No rejected endpoint is promoted into a result.

## Native time refinement

`finite-glue-plastic-refine-time-001` preserves the 0.25-degree load and 3.6 ms
physical protocol while refining 100 µs steps to 25 µs: 32 load, 16 hold,
32 unload and 64 constrained settling steps. All 144 steps were accepted in
both environments. The loaded arm had nonzero plastic strain; the reference
had exactly zero in all six plastic-strain components. At the final constrained
state, maximum absolute plastic strain was 0.00289957, with 563 cells above
1e-6. This is retained material history, not proof of a free residual fold.
Final free-node displacement was 1.9804 µm, speed 0.6121 mm/s and kinetic energy
8.386e-12 J, so the endpoint is not asserted to be equilibrium.

`finite-glue-elastic-002` separately completed the earlier 36-step cycle with
matching elastic-control constants and no plastic state. Its residual motion
must not be called plastic strain. The final rebuilt binary then completed
three matching 144-step protocols, all with successful status and computed
acceptance for all 288 environment observations per run:

| Final run | Load | Maximum final absolute plastic strain | Cells above 1e-6 |
|---|---:|---:|---:|
| `finite-glue-default-final-001` | 0.25° | 0.00289957 | 563 |
| `finite-glue-elastic-final-001` | 0.25° | zero by the stateless elastic law | 0 |
| `finite-glue-unused-angle-final-001` | 0.125° | 0.00192611 | 368 |

Both plastic runs kept the held reference at exactly zero plastic strain. The
smaller load produced less final plastic strain under this model. The elastic
control still had 1.971 µm displacement and 0.460 mm/s speed at the endpoint,
showing why endpoint displacement alone cannot identify plasticity. No endpoint
is claimed to be equilibrium. The separate exported-geometry audits passed
for both final plastic runs. [The comparison](final-cycle-comparison.json) binds
its input bytes and includes per-material and phase-end observations;
`summarize_final_cycles.py` reproduces it from the full raw run outputs at the recorded evidence root.
All nine targeted compiler/Metal regression checks passed.

The default and half-angle native runs used the same canonical materials,
rebuilt probe and production metallib, with unchanged input hashes throughout
each run. The model sources and binary identities are in each invocation receipt.

The early CSV name `certificate_accepted` copied a raw certificate field that
can remain one when the global step is rejected. Successful status and actual
residual gates are both required. The final probe exports the raw field as
`certificate_raw_accepted_flag` and a separate computed `step_accepted`.
Failure snapshots are rolled-back accepted physical state, not the rejected
candidate geometry. Do not treat raw stress scratch-state changes as plasticity.

## Provenance and limits

`retained-files.json` lists original paths and hashes; compressed source inputs
and phase-end material states preserve exact bytes on decompression. Full OBJ
sequences, packages and invocation receipts remain outside the checkout under
`/Users/home/cardboard-evidence-20261006`. Development source receipts can
include edits newer than the binary under test; the final default replay is
built after the final probe changes and is the final implementation check.

Mesh convergence, a temporal limit, quasi-static behavior, creasing-tool
contact, adhesive/paper failure, published crease-curve agreement, released-grip
shape and full-box folding/assembly remain unqualified. A smaller timestep
completing this cycle is numerical progress, not proof that every fold works.
