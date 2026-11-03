# Align the FEM contact potential with its certified clearance

The exact root-2 / iteration-12 witness is skin self-contact 2052768 (nodes 61202, 61204, 61325, 61213). Its immutable-source point/triangle plane gap is about 1.792 nm. During Newton, the gap approaches the 1 nm hard certificate margin. The previous IPC potential was unshifted and remained finite there; fallback projected CCD used an even larger 3.194 nm floor and returned time zero. This is an inconsistent numerical contact domain, not proof that the anatomical surfaces intersect.

NM_FEM_CLEARANCE_ALIGNED_BARRIER=1 is an opt-in correction for eligible immutable reference FEM self-contact. The existing positive, source-certified dyadic clearance is stored in an unused internal Hessian-row metadata lane. Potential evaluation uses separation minus clearance and activation width minus clearance. The force and stored PSD curvature differentiate that same translated IPC potential. No collision, determinant, residual, mass or rollback acceptance limit is relaxed; no new material, geometry or simulator is introduced.

The aligned response uses dimensionless IPC derivatives to avoid fourth-power underflow at nanometre gaps. Legacy evaluation with the option disabled remains unchanged. An unshifted-only exact VT derivative explicitly falls back to the stored consistent PSD metric for shifted contacts.

Validation:
- 34 GPU arithmetic cases compare potential, force and curvature against independent Float64 formulas; representative forces also pass independent energy finite differences. Nanometre, micrometre and centimetre scales, unchanged activation, below-floor C1 continuation, invalid shifted domain and output canaries are covered.
- The legacy native suite passes 798,961 checks; graph cache81 parity/guard checks pass 1,919.
- The full native trial is reported separately after completion. Kernel checks do not qualify the integrated scene or throughput.
- Early trial 007 was stopped during preparation because its harness stage expected 3 steps while the native target was 16. The replacement 008 pins matched 16-step settings.

This changes the numerical contact potential to agree with its existing geometric domain. It is not a bitwise-equivalent algorithm optimization. Integrated residual, contact, tissue, accepted-state and throughput results govern promotion.

Integrated result, pinned trial 008:
- Accepted 4 steps (62.5 microseconds) before a later code-10 rejection. The matched thin-cell experiment without alignment accepted 2. The retained baseline without the experimental preconditioner remains better at 9 steps, so this is not an overall best-scene qualification.
- GPU step times were 20.333, 19.869, 33.546 and 22.569 seconds. Whole command elapsed time including preparation and the rejected attempt was 392.075 seconds. No throughput gain or 0.1x result is claimed.
- The former root-2 zero-bound point passed; root 4 still rejects. The mechanism is not fully resolved and the option remains off by default.

The follow-up adds NM_FEM_DEFORMABLE_LIMIT_TRACE_FAILURES=1: bounded, read-only minimum-producing records on all 128 production tiles, with unique tile/lane ownership. It logs zero-bound contacts directly on failure, including compensated features and the stored numerical offset. It does not serialize contact processing or make CPU physical decisions. Debug readbacks are explicitly instrumented, bounded to 16 MiB per snapshot, and are not an uninstrumented throughput measurement.
The latest native regression suite passes 810,084 checks, including full-record lane ownership/canaries and exact physical-bound parity. The aligned energy/force/curvature tests now run in the default suite. A separate full-scene automatic-witness run is pending; this checkpoint does not claim it passed.
