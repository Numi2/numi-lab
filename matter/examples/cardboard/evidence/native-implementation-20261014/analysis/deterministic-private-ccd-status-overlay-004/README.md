# Deterministic private deformable narrowphase status selection — overlay 004

This versioned candidate is based on source109 and preserves the 003 runtime and Metal changes. It adds a bounded control to the already-registered native CCD checker; no production physics, tolerance, capacity, or solver code changes from 003 are introduced.

The existing checks cover production private-status clearing and lowest-slot reduction. The new native control dispatches `nm_contact_narrowphase_deformable` on a single active point-point candidate. Its FP32 positions (`+/-1e20`) and candidate velocities (`-/+2e20`) are finite, with `dt=1`; the relative squared terms overflow, so the production `pointPointCCD` path returns uncertified and reaches `recordDeformableNarrowphaseFailure`. This is a deliberately extreme status-routing witness, not a physically meaningful contact case.

With deterministic selector 1, the checker requires the producer to write `(CONTACT_FAILURE, object=0, slot=0, diagnostics=[0,1,0,0])` to that candidate's status, preserve the environment's success record before reduction, and then have the production reducer propagate the same tuple while preserving counters. It repeats the same geometric branch with selector 0 and requires the environment status to receive that tuple while the candidate-status sentinel remains byte-identical. These controls exercise status routing only; physical qualification is false.

The existing ordinary path retains first-writer status behavior. Private status storage remains `environmentCount * deformableContactCapacity * sizeof(NMMatterStatusGPU)` and continues to use the checked runtime allocation from overlay003. The new fixture uses only fixed-size, single-environment buffers.

`contract_test.py` passes 18 CPU source/contract checks. This overlay was prepared without Metal compilation or native GPU execution; root should compile and run the existing native checker target before claiming the geometric control has passed.
