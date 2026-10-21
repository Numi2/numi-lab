# Build 169: integration results and owner correction

All native work ran on SSH Numi. The optional changes remain off by default. No visual verification was performed.

## A real passing integration increment

The embedded Human now admits the existing operator ICB: the lifted input is included in the declared resources and the hybrid-mode bit is included in graph-cache identity. Direct versus ICB execution produces exactly equal 17 audit values, 14,202 node records and 54,133 tetrahedron records. The preconditioner prefix ICB remains unavailable for hybrid coordinates; its separate admission guard is retained. This is operator-execution parity, not a passing reduced-model accuracy qualification.

Sequential refinement preserves prior fine rows and enforces the remaining fine-coordinate budget. Its CPU checker passes. A native 20-row refinement attempt fails before execution because selected optic rows overlap the protected optic-boundary policy. That failure is retained and is being corrected by recognizing already-protected fine rows without removing their protection.

## Correction to the earlier indirect-dispatch measurements

The native resting scene uses NumiHumanRestingCoupling -> TransactionSourceMassQuery -> MetalArticulatedOperator. It does not use the MetalNumanXHumanMatter adapter for this action. That actual transaction bridge did not advertise the newly added indirect/borrowed capabilities.

Consequently the build165 flag pairs did NOT establish execution of indirect mass dispatch in the native Human, and the build169 borrowed-encoder pairs did NOT exercise the borrowed route. Their small timing differences are controls, not optimization gains. Original evidence and reports remain retained; this correction supersedes any implication that those native pairs exercised the new feature. The isolated GPU argument fixture demonstrated only its stated fixture behavior.

Further measurement requires wiring the actual consumer and observing the runtime route marker. Capability and transaction-lease checks remain mandatory.

## Measurements that change the priorities

- Host encoder submission instrumentation totals approximately 0.068 seconds for the first root, mostly linear work (0.056 seconds). It does not explain several seconds of wall time; it introduces no GPU encoder splits or waits.
- Jacobi eight-sweep telemetry: 1,508,910 attempts, 1,364,362 accepted, 144,548 refined-LU fallbacks. Fixed dense sweeps are costly even when most attempts pass; fallback alone does not explain the regression.
- Three accepted initial roots: first root 1,587 used columns, second 345, third 83; each still encodes 384 restart cycles. This strengthens the case for suppressing the complete inactive graph.
- The two post-first-root intervals advance 0.000250000012 seconds in 7.34138525 wall seconds: 0.0000340535x. This excludes initial load/compile/parity and first observer, but is only a short initialization sequence, not settled throughput.
- Assembled SIMD-row operator: physical root accepted, body 10.192490 seconds; 1% audit FAIL (maximum 8.15065%). Not promoted.
- Conservative Morton-order cache: physical root accepted, all audit metrics exact, body 9.589834 seconds. Single pair only; no sustained speed claim.

The supported equilibrium checkpoint, accuracy-qualified large timestep, complete graph elision, reduced-model force/strain fidelity, and 0.1x integrated qualification remain open.

## Actual owner / default-layout correction

The default Human Krylov layout contains 28,450 slots: 28,404 FEM slots and 46 vascular slots, with zero rigid/support slots. Therefore the default native benchmarks do not exercise coupled source-mass products. Borrowed/indirect source-mass flags in those runs cannot establish performance of that path. Subsequent actual-owner experiments must use the declared exact-candidate / implicit-body-spring configuration and verify route telemetry; their different system must not be compared as equivalent to the default system. Existing failed/inactive trials remain retained.
