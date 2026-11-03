# Active cavity and completed contact execution, 2026-11-03

Two default-off internal execution policies retain the current material equations, precision, physical state and final acceptance gates.

- NM_VASCULAR_COOPERATIVE_CAVITY_OPERATOR=1 shares compensated cavity face evaluation across one SIMD group per equation. Face sums retain the original order. Direct and cached operator ICB paths select the same implementation.
- NM_CONTACT_COMPLETED_INCIDENCE_REUSE=1 suppresses main-candidate incidence rebuilding only after the existing immutable completed-geometry admission, a finite nonlinear convergence latch, healthy status and exact-zero geometry proof. It reuses the existing CSR with one 16-byte transient indirect-dispatch buffer. Fresh friction, trial work and final full narrowphase/incidence/residual/certification remain active.

Validation: the existing 916,531-assertion solver contract, 189 focused contact-dispatch assertions and 36 cooperative JVP cases passed. Valid JVP outputs match bitwise; invalid geometry is rejected. Two environments, multiple face blocks, low words, signed/cancellation directions, completed/failed and NaN gates are covered.

The final combined native binary accepted four steps with 32 active Krylov columns and two active Newton begins per step. The complete captured endpoint (833,232,924 bytes) is bitwise equal to completed-linear-builders-063: SHA-256 66c476e14b867eb9a19039ec112750a3d517a18ce039b24bdb5181fec7ac24c7. Physiology CSV and accepted coordinate artifacts also match bitwise.

Exact source bundles, source/binary pins, configuration, device metadata, launch command, traces and status receipts are retained here. Matched quiet performance repetitions are still running; no repeatable speedup is claimed in this increment.

This is a 62.5-microsecond initialization prefix, not supported motion, sustained active-Brain throughput or native presentation qualification. The 0.1x target remains unmet. The earlier resource-stopped capture is retained on Numi and will be included with the complete timing report; its partial endpoint is unqualified.
