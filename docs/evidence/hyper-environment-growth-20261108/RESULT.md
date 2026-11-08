# HyperPolicy environment-growth continuity

**Fixed:** expanding a serialized HyperPolicy runtime from one environment to two no longer discards the existing row of phase and previous physical-action state. A host-side rejection after encoding the growth but before command submission also preserves that state on retry.

The preregistered Apple M6 test uses a deterministic, untrained policy with 20 ms physical steps. Because a MetalWorld resident has a fixed environment count, the two calls use separate residents; the second call deliberately supplies a policy-only mask to model one continuing logical row and one new row. The world retains its ordinary initial reset. This isolates the action-owner contract and is not a continuous physical episode.

| Case | Existing row, phase/action after second call | New or reset row |
| --- | --- | --- |
| Baseline, 1 to 2 environments | 0.02 / 0.005 | 0.02 / 0.005 |
| Repaired, 1 to 2 environments | 0.04 / 0.010 | 0.02 / 0.005 |
| Repaired, growth rejected before commit then retried | 0.04 / 0.010 | 0.02 / 0.005 |
| Preallocated two-environment control | 0.04 / 0.010 | 0.02 / 0.005 after reset |

All physical steps in this fixture succeeded. Both repetitions matched. A supplemental control omitting the new row reset verifies the newly allocated row begins at 0.02/0.005; the already active second row of the preallocated control instead continues at 0.04/0.010. The initial test incorrectly demanded those two arms match and failed; INITIALIZATION-AMENDMENT.md, final-ctest-2.log and the unmodified outputs retain that error and the correction.

Growth now allocates a complete replacement set before installing it, zeros the new buffers and copies the old live phase and action prefixes on the borrowed command buffer. The previous allocation and growth command remain retained until a later serialized owner call determines whether that command was submitted; an abandoned uncommitted command restores the old allocation before retry. Rollout trace reads also settle this provisional state. No separate queue, synchronous wait, or CPU readback was added. Allocation failure leaves the prior allocation installed by construction, but allocation failure was not fault-injected.

The permanent hyper_policy_environment_growth regression covers growing versus preallocated capacity, host abort and retry, explicit reset, unreset new-row initialization and repeatability. On both the combined checkout and the independently built main-source checkout, it passed alongside the existing HyperPolicy physical-rejection, status-restore, control-step-capacity and runtime-program checks: 5/5 each. The two builds produced identical captured phase/action JSON in both mask modes; binary identities are recorded separately in final-sha256.txt and main-sha256.txt. The initial build request named nonexistent targets and failed; candidate-build.log retains it. An initial hash command used an incorrect relative path and was rerun with the corrected path.

This repair requires the existing serialized owner contract and stable logical environment-row identity. It does not qualify overlapping policy submissions, mapping arbitrary environments between owners, a full physical no-reset retry, an enqueued-but-never-committed command, partially executed GPU-command failure, recurrent training, force accuracy or learned skill. The retained stateless masked-depth integration is a separate test and was not repeated with this policy. Both experimental directional CCD and augmented-Lagrangian contact remain disabled.

Reproduction uses the source-controlled tests and commands.txt. Baseline and final native outputs, plans, failed checks, compiler logs and both build identities remain in this directory.
