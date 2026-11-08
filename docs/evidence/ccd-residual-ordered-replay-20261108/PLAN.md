# Residual-triggered ordered contact replay: isolated diagnostic

Before execution, build two isolated research shaders: the current main
Wave32 solver and the previously retained four-contact research solver, each
modified to request the existing ordered rigid-contact replay when an island
has no rod and the reported normal residual exceeds 0.002 m/s. Reuse the
existing replay iteration configuration without changing any iteration
budgets. The replay is a different sequential update, not a claim that simply
running more fixed Jacobi sweeps repairs contact. The flag is an experimental
probe, not an acceptance certificate or production change. The current replay
is environment-wide and skips rod/generalized rows; do not infer mixed-island
safety from a no-rod local condition.

Link both diagnostic solvers to the separate unpromoted monotonic box-plane
certificate from commit 3bf1e510b, with both original experimental selectors
off. Run the same 12 hybrid fixtures at production CCD cap 16, identical
initial states, material, solver settings and runtime; retain trace on/off,
repeat, exposed restoration, impulse and independently reconstructed normal
residuals. Run the existing accepted-segment geometry observer. Compare to the
matching retained unmodified solver controls. Near misses must remain force
free; impacts must complete; normal residuals must be <=0.002 m/s. Check
contact response rather than accepting reduced iterations as evidence.

If the targeted gates pass, repeat against current-runtime controls on the
100-step impact, force-free near miss, supported and sliding box paths and the
matched mixed rod-rigid cases, with friction, q/v, contact normal residual,
work and restoration checked where observable. A candidate that silently
leaves residuals above the gate or corrupts friction or mixed coupling is not
promotable. The certificate and four-contact candidate themselves remain
unqualified. Do not report timing as a performance improvement. Preserve every
negative result and do not modify production code.
