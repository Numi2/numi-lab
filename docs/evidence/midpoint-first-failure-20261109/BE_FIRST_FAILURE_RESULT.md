# Backward Euler precontact first-failure trace

The bounded trace reproduced all 3,186 accepted steps and the subsequent
failure from the untraced 1 microsecond run. All recorded physical trace
fields except GPU timing matched, the restored physical-state hash matched,
and the 16 selected Newton iteration records repeated exactly during replay.
The step is before contact; no contact sample was active in the evaluated
capture, and the minimum node gap in that candidate was 150.164 micrometres.

The solver took the full line-search length on all 16 Newton iterations. Its
reported nonlinear residual decreased from 6.43408e-5 N s to approximately
2.69792e-12 N s, then stagnated. From iteration 3 onward all 27 nonzero
vertical velocity corrections rounded away when added to their stored FP32
values; later iterations also lost horizontal corrections. This was checked
against the captured pre-apply working nodes and corrections and the actual
FP32 addition. It identifies a representability limitation at this state; it
does not by itself prove that any proposed precision change completes a
rollout or qualifies its forces.

An independent FP64 evaluation of the same captured backward-Euler equations
returned a residual norm of 2.69743e-12 N s. The captured GPU residual had
norm 2.69792e-12 N s and differed from the independently evaluated residual
vector by 2.23e-15 N s, about 0.083% of its norm. Both exceed the configured
physical residual limit of 2.24813e-12 N s. The independent maximum local
scaled residual without a unit floor was 0.0001340; this is a separate measure,
not a replacement for the failed overall certificate. No tolerance or solver
budget was relaxed.

The complete trace, GPU arenas, diagnostic source, exact offline evaluator
hashes, analysis and execution metadata are retained. This diagnostic is not a
native accepted step, a backward Euler force-accuracy result or a timing
comparison. Both experimental contact and collision options remain disabled.
