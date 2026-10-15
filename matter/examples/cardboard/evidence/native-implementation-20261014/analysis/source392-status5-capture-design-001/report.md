# Source 392 status-5 capture design

## Scope and evidence binding

This is a read-only instrumentation design for the first `NM_STATUS_INVALID_DEFORMATION` event in the source-392 failed transaction. It does not identify the event's producer from the existing capture and does not claim that tet 858's constitutive projection caused the failure.

Bound inputs:

- Source snapshot: `source-scaled-material-merit-118`.
- `matter/src/metal/common.metalinc` SHA-256: `66153b6ac14d361e33f9f74d228b8882db81bb35f6e425c973c206bd196a5f55`.
- `matter/src/metal/fem.metalinc` SHA-256: `742cf561ee1ed91e20ed797e664595f675302437a3d583f8c5768da6cb3d008e`.
- `matter/src/runtime.mm` SHA-256: `3c467b04e72512b53a5ba9e1f3ee4132f293b0d2e6ac22acc0c3936512319e74`.
- Source-392 probe manifest SHA-256: `fbfcf51db84a3710cc3bd96e016597728163da464974ea033554e82e587cd8eb`.
- Source-392 external source-build binding SHA-256: `da94e6051e767ea4bfba58705ccaf321e5dd97e5e9b40b32715f056676980c9f`.
- Existing failed output reports zero accepted steps, environment 0 status 5, object 0, failing index 858, diagnostics `(0.9977478981, 0, 0, 0)`. This tuple is not a source-site identifier.
- The package manifest declares the corrugator starch interface disabled with zero paired faces; the run has no MPM material object. This excludes cohesive-face and MPM writers for this particular transaction only, subject to the compiled-world binding.

## Status-5 producer inventory

`NM_STATUS_INVALID_DEFORMATION` is status value 5 (`matter/include/numi/matter/shared.h:139`). Literal status-5 producers in the bound Metal source are:

| Kernel / source | Condition and payload | Relevance to source 392 |
|---|---|---|
| `nm_fem_apply_kinematic_targets` (`fem.metalinc:817`) | Non-finite derived displacement/velocity; object index is `NM_INVALID_INDEX`, failing index is a node. | Monitor if this kernel is active. The observed object 0 / index 858 tuple is inconsistent with this payload shape. |
| `nm_fem_internal_forces` (`fem.metalinc:1097`) | Invalid constitutive stress evaluation or stress above `material.validity.z`; failing index is the tetrahedron; diagnostics are `(detF, maxAbs(stress), 0, 0)`. Invalid local evaluation can leave stress zero, so the observed zero second component is compatible but not proof. | Primary candidate producer for object 0 / tet 858. |
| `nm_fem_cohesive_face_forces` (`fem.metalinc:1803–2015`) | Invalid face/object/frame/traction evaluation; failing index is local face. | Inactive for this compiled world if its zero face count is confirmed. |
| `nm_fem_cohesive_face_operator` (`fem.metalinc:2264–2460`) | Invalid cohesive tangent/geometry evaluation; failing index is local face. | Inactive for this compiled world if its zero face count is confirmed. |
| `nm_fem_prepare_cohesive_activation` (`fem.metalinc:2637`) | Invalid cohesive-face frame. | Inactive for this compiled world if its zero face count is confirmed. |
| `nm_fem_armijo_validate_geometry` (`fem.metalinc:4402`) | Candidate determinant outside its admissible interval; object 0, failing index is `microstep.solverIteration` (not a tetrahedron ID), diagnostics are `(detF, validityLower, validityUpper, numericalFloor)`. | Must be distinguished from a tet-indexed failure. |
| `nm_fem_validate` (`fem.metalinc:4725`) | Invalid determinant or `evaluateMaterialValidity`/validity margin; failing index is tetrahedron; diagnostics are `(detF, validityMargin, 0, 0)`. | Another possible producer for an object 0 / tet 858 failure. |
| `nm_mpm_g2p` (`mpm.metalinc:1055`) | Invalid particle update; failing index is particle. | Inactive only if the compiled world confirms zero MPM particles. |

`recordFailure` (`common.metalinc:295–319`) atomically claims `status.code` from success, then writes object/index/diagnostics. It does not record a kernel/site ID. The winner is therefore already deterministic only as to the status tuple's first successful CAS; the tuple does not tell whether `internal_forces` or `validate` produced it.

Relevant runtime dispatch sites include kinematic target application at `runtime.mm:14043`; base/internal-force evaluation at `17531` and `22461`; Armijo geometry/internal force calls around `11771/11787`, `20991/21004`, and `21531/21619`; and final FEM validation at `22769`. Each active status-5 producer in the selected path must have an observer immediately after its dispatch. Do not infer the site solely from `failingIndex == 858`.

## Smallest safe capture

Add an opt-in, read-only `nm_fem_status5_capture` observer kernel and a separate bounded diagnostic buffer. Do not change constitutive equations, `recordFailure`, candidate buffers, force buffers, solver statuses, accepted histories, rollback, or line-search decisions.

After each active status-5-producing dispatch, issue a buffer barrier and the observer with a host-authored stage ID plus that dispatch's context. The observer reads the completed status and writes only the diagnostic buffer when `status.code == NM_STATUS_INVALID_DEFORMATION`. It captures one event slot per environment and status-reset epoch, not one entry per failing tetrahedron thread. The runtime assigns unique event slots from control-step, microtick, Newton iteration, stage, and Armijo trial; it precomputes the required capacity with checked arithmetic. If capacity or schema bounds fail, mark the diagnostic incomplete and leave the physical transaction running unchanged. Never silently overwrite or truncate an event.

For tet-indexed FEM status events, capture:

- exact `NMMatterStatusGPU` tuple, stage ID, status-buffer domain, environment, control step, full `NMMicrostepGPU`, Newton/trial index, and selected Armijo alpha when applicable;
- the compiled `NMTetrahedronGPU` and its material/object IDs;
- all four exact accepted and candidate `NMFEMNodeStateGPU` words (the center and trial candidate separately when this is an Armijo trial);
- the exact accepted internal-state words for this tet and the `stateCount`/material descriptor;
- production-computed deformation high/low words and deformation-rate input, plus timestep and the material parameter slice used by the material program;
- capture completeness, event ordinal, and source/device/world fingerprints.

For geometry-gate status, preserve the kernel's `failingIndex` semantics and capture the offending tet index explicitly. `nm_fem_armijo_validate_geometry` currently stores solver iteration there rather than the tet that tripped the determinant gate; a sidecar must not relabel it as a tetrahedron. Its observer can carry a separate diagnostic offender field supplied by a minimal read-only witness output from that kernel, or report offender-unavailable. Do not scan and guess after the fact.

The status observer must run before rollback or any buffer reinitialization. Its record is diagnostic-only; do not bind it as a solver input or alter production status. The native point material replay can then be run on the captured exact candidate and accepted history, but only after the recorded stage confirms that the event came from a constitutive kernel.

## Parity and admission checks

Run diagnostic-off and diagnostic-on transactions from the same immutable source, compiled world, binary/material package, seed, and command. Compare every existing physical output byte-for-byte: final result/status, observation CSV, candidate/rejected geometry artifacts, accepted histories, transaction/checkpoint files, and source-independent numerical traces. The diagnostic buffer is the only permitted extra output. Require the same terminal status tuple and exact accepted/rejected physical state. The extra observer dispatches are downstream of existing producer kernels and may not write any existing buffer.

A source-site report is conclusive only if (1) the exact compiled-world hash confirms zero cohesive faces/MPM particles for this run, (2) every active FEM status-5 producer was followed by an observer before another status reset or rollback, (3) the event record has intact node/history/F words and no overflow, and (4) its status tuple matches the retained final failure. Otherwise label the result incomplete and retain the old ambiguous status-5 evidence.

No native capture was run for this design. The report is not a material or physical qualification result.
