# Bounded hard-equality and scalar-contact experiment

The Metal quality solver includes a bounded exact active-set path for independent
hard scalar equality rows and exactly one compliant unilateral scalar normal.
It enumerates inactive and active contact, solves the corresponding Schur
systems and audits momentum, hard equalities and regularized complementarity
before publication. Singular systems or failed audits reject transactionally.
The path is limited to at most 96 generalized velocities and at most as many
blocks as velocities. Unsupported products use the existing solver path.

`metalrobo.crow.quality_mixed_scalar_analytic` tests closed-form active,
inactive and coupled-mass cases, block reordering, inconsistent hard rows,
objective reporting and exclusion of a two-contact problem.
`metalrobo.crow.quality_mixed_scalar_random` checks 192 deterministic synthetic
cases against independent FP64 KKT references. The reference inputs are
float32, matching the native solve. The research evidence and failed baseline
are archived in the crow evidence branch:
https://github.com/Numi2/numi-gpu-physics-pigeon-sim/tree/agent/crow-neural-balance-evidence-20261115/evidence/crow-feather-rachis-20261116/mixed-single-contact-candidate-20261116

This is not a friction-cone or multi-contact solver qualification, a live
articulated feather attachment, or a full world integration. It currently runs
after the generic solve and has not been performance-qualified. Do not infer
flight, landing or biological realism from these tests.
