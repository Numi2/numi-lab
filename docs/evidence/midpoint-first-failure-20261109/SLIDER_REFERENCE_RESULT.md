# Independent reacting-slider reference

The independently evaluated n=2 cube and 10 g vertical slider completed the
10 ms case at both 0.5 and 0.25 microseconds. The unchanged reference area was
0.0004 m², minimum relative gap was 94.5373 micrometres, and contact first
appeared at approximately 5 ms. The finer run produced 0.211598 N sampled
peak force, 7.599275e-5 N s integrated normal impulse and final slider
position -0.000324173 m and speed -0.0856993 m/s. Its integrated FEM contact
moment about the fixed origin was (+6.888656e-7, -6.888656e-7, 0) N m s.
The equal-and-opposite slider contact moment is enforced by this reference
model, with a zero-work orientation constraint. This is not an independent
measurement of a native solver rigid reaction.

The temporal comparisons pass the registered force, impulse and body-motion
gates; force-bin and sampled-peak differences are separately recorded in
`slider-reference-analysis.json`. The largest vertical momentum balance error
was 1.10e-16 N s and total-energy drift was 1.06e-14 relative, including FEM
and slider kinetic and gravitational energy, elastic energy and barrier
potential. Relative interface contact work was -2.838e-10 J. These are
numerical checks of the reference equations and do not establish material or
spatial accuracy.

The source, exact executable hash, trajectories, and analysis are retained;
the executable is copied under `evidence-slider-reference` in the native build
directory. This independent CPU trajectory is a validation target only. It
is not a native two-way midpoint transaction, a rotating-body test, or a
whole-environment rollback/retry result. The production guard remains active.
