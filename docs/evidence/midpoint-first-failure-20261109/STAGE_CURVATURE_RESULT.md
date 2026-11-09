# Local independent linearization check

As a follow-up to the preregistered single-correction probe, the same independent
FP64 equations were evaluated at correction scales 0, 0.5, 1 and 2. The norm
of the discrete second difference over the full interval was 1.10e-21 N s,
small compared with the resulting 6.304e-15 N s full-step residual. Thus the
finite-difference secant provides a useful local approximation to the tangent
for this captured direction, subject to its own arithmetic error; this is not
an analytic tangent proof. Half the independent full-step residual at scale 1
is 3.152e-15 N s, about 1.7% above the solver-reported 3.099e-15 linear metric
at iteration 3. The independently evaluated endpoint is unrounded in the
midpoint stage, while the actual native candidate did not apply most vertical
corrections. No native contact or full rollout is certified by this diagnostic.

The source and input hashes and all sampled residual norms are retained. The
14 default tests passed after the trace-only runtime change. This check is
post-hoc and does not change the fixed physical acceptance limit.
