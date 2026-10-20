# Exact constitutive tangent specialization
Preregistered before native experiments. Parent ae35df81. SSH numi only; one GPU job at a time. No visual verification.

Hypothesis: structurally certified stateless compressible neo-Hookean materials can evaluate their exact directional tangent without the generic scalar-expression interpreter. No material parameter, energy, force, timestep, precision, physical state or acceptance tolerance changes.

Eligibility must be compiler-derived from exact AST and parameter identity, exclude state/dissipation/rate/fiber/other energies, and retain generic fallback. Default runtime clears eligibility; NM_FEM_CANONICAL_NEOHOOKEAN_TANGENT=1 opts in. Native differential/finite-difference and negative classifier tests precede Human benchmarking.

A/B uses the same freshly linked binary and assets, only this flag differs, retaining distributed CGS2/ICB, radix contact, distributed global measurements, and preconditioner reuse. Four roots at dt 0.125 ms (0.5 ms simulated), three pairs in alternating order. Prediction: median body throughput improves at least 1.5x; report failure if not. Numerical gate max(1e-7,1% relative) on existing endpoint metrics, unchanged tissue/contact acceptance; no long run from a failing candidate. Profile every tangent-cache invocation in root3 plus whole-stage spans and compare profiling overhead to uninstrumented runs. No sampled-cost extrapolation.

This does not qualify supported rest, integrated long-run throughput or 0.1x. Failed trials are retained. Follow-up residual-adaptive Newton forcing is a separate mathematical experiment, with unchanged full residual/vascular bounds.
