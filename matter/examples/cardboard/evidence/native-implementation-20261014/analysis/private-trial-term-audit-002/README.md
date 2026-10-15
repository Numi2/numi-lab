# Private Armijo trial term audit, capture 312

This is an observational replay of one rejected native trial, not a physical or solver-convergence qualification. It checks the residual-gather portion of the private trial using the exact captured force/sample records and production incidence ordering.

## Bound evidence

- Source snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-contact-capture-and-resolved-material-sweep-095`.
- Capture manifest: `/Users/home/cardboard-implementation-20261014/native/corrected-private-trial-layout-312/private-trial.coupled.json` (SHA-256 `fea8776aa57e3a1e11a2720b8a4acd747c3d30e19a1c85fafebdf43794cb1477`).
- Adjacent payload: `/Users/home/cardboard-implementation-20261014/native/corrected-private-trial-layout-312/private-trial.coupled.bin` (SHA-256 `58206c37aa9cd897d490a7dd45979d887b0e2f44992fefad512e943fb10d17ff`, 19,467,248 bytes, 129 arrays).
- Selected point: root 1201, Newton iteration 12, trial 22; both pre/post rebuild capture phases. `physical_validation=false`, `accepted_checkpoint=false`; the enclosing run exits 1.
- Source file hashes and capture identity are retained in `case312-report.json` (manifest SHA `fea8776aa57e3a1e11a2720b8a4acd747c3d30e19a1c85fafebdf43794cb1477`; payload SHA `58206c37aa9cd897d490a7dd45979d887b0e2f44992fefad512e943fb10d17ff`).
- Capture-on/off parity: `/Users/home/cardboard-implementation-20261014/analysis/private-trial-complete-capture-parity-312-313-001/report.json` (SHA-256 `e12d484a6d750b75cc68204903b2103d09bb6bc2c67729df2abece878d3dfae5`); all 11 non-log production files and all 32 Armijo records are byte-identical. Both executions reject. This is instrumentation noninterference for this failed prefix only.

## Exact gather result

`audit_trial_terms.py` verifies, at captured FP32 word level and with the observed incidence order/FMA behavior, that each reconstructed contact/cohesive residual gather reproduces the captured cumulative FEM xyz residual checkpoint bit-for-bit:

- Cohesive: 1 face, 6 node incidences, zero invalid map entries.
- Rigid contact: 5 valid samples, 72 node incidence slots, zero invalid map entries.
- Deformable contact: 100 active records, 402 node-incidence visits, zero invalid map entries.

The reconstructed final metric is 3.4360307180e-6; the captured Armijo trial norm is 3.4360307382e-6. They differ by about 2.0e-14, consistent with the captured GPU reduction being FP32 while the audit norm is accumulated in FP64. FEM free xyz contributes 2.8629662660e-6 and the final rebuilt rigid generalized x rows contribute 1.8999292763e-6.

## Validity limits

- Exact bitwise equality establishes that captured force records and incidence gathers reproduce these trial residual checkpoints. It does not independently validate broadphase, narrowphase, CCD, contact constitutive response, history evolution, cohesive law evaluation, or the production Jacobian.
- Residual stages are cumulative FP32 checkpoints. Reported term magnitudes from their differences are observational, not separate independent force calculations.
- Bulk residual is not independently reconstructed because this capture does not include the full FEM element-to-node incidence order.
- The rigid generalized rows in the checkpoint before free-body assembly are stale pre-clear scratch. Only the final rebuilt rigid block is meaningful; no subtraction of the stale block is interpreted as a physical term.
- The selected capture is a rejected trial/run. It is not an accepted transaction, physical validation, or evidence of nonlinear convergence.
- History hashes differ between transaction-accepted arrays and private-trial inputs at Newton 12. This alone does not establish mutation; the private solver may use current Newton-accepted working history. The capture does not provide a matching center-operator generation token to disambiguate that lineage.

## Newton derivative comparison availability

The capture does include the 78-lane `fem_solution_correction`, the Newton linearization nodes, trial candidate nodes, and trial residual checkpoints. It does **not** include the full Newton-center residual vector or a production `J * delta` vector, and it has no serialized explicit row-index map binding all correction/residual lanes to the constrained FEM and rigid generalized rows. The Armijo records contain scalar norms only. Therefore the existing 129 arrays cannot establish or localize `R(x_k + alpha delta) - R(x_k) - alpha J(x_k) delta`; correction/alpha consistency and trial gather parity are not substitutes for that derivative test.

A follow-up capture should export the center residual and production operator action from the same accepted linearization, along with the exact row map and generation identity for material/contact histories and active contact branches. Keep pre-rebuild and post-rebuild trial candidates separate: they are both trial geometry, and in capture 312 their node records are bitwise identical.

## Reproduction

```sh
python3 -m py_compile /Users/home/cardboard-implementation-20261014/analysis/private-trial-term-audit-002/audit_trial_terms.py
python3 /Users/home/cardboard-implementation-20261014/analysis/private-trial-term-audit-002/audit_trial_terms.py /Users/home/cardboard-implementation-20261014/native/corrected-private-trial-layout-312/private-trial.coupled.json --out /Users/home/cardboard-implementation-20261014/analysis/private-trial-term-audit-002/case312-report.json
```

`audit_trial_terms.py` SHA-256: `6c700979b71222d7839a56fb29ca36bfef84fdcc20e8f0edccead07e4215ea5e`.
`case312-report.json` SHA-256: `3034ffc64425a67e8d1bcac7a321dc0195284c6f48f7c9dae3438966381b58ae`.
