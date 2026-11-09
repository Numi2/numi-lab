# Bounded-memory FEM reference preparation

The 64-to-128 coarse spatial comparison still has a 20.7% sampled-peak
force difference. The 128-division fine temporal reference is running; its
result is required before treating that comparison as a valid spatial test.

A separate FP64 structured-mesh evaluator was prepared for a possible next
mesh. It keeps the same material, contact potential, quadrature, initial
conditions, and classical RK4 equations, but recomputes element data and uses
eight cell colors to avoid retaining a per-element force arena. Cells of one
color have disjoint vertices. The implementation changes summation order and
therefore needs numerical comparison; it is not assumed equivalent.

On deformed n=2, 4 and 8 fixtures, the independent serial acceleration
comparison passed at 1.19e-17 relative maximum; mass and contact weights
were identical. In complete 10 ms n=2 and n=8 trajectories at h=2 µs,
compared with the frozen ordered evaluator, the largest force-trace difference
was 2.22e-14 of the reference peak and the cumulative-impulse difference was
below 1e-15 of total reference impulse. See
`compact-reference-trajectory-parity-006.json` and its four retained traces;
`compact-reference-ctest-003.log` records the deformed-state test. This parity
only covers those small meshes and timestep.

A guarded n=256 model construction and one full noncontact RK4 step passed
its mass, area and free-fall checks with a self-reported peak
resident set of 4,755,046,400 bytes, below the 6 GiB guard. The exact binary
hash, command, limits and result are in
`compact-reference-n256-rk4-capacity-002-execution.json`. This does not
qualify the contact trajectory, temporal convergence, spatial convergence,
or production performance. The earlier one-RHS capacity run
`compact-reference-n256-capacity-001-execution.json` retained an unreliable
external RSS sample that only saw startup; the repeat
`compact-reference-n256-capacity-002-execution.json` captured the process
high-water itself. Neither one-RHS run establishes full-step memory.

No n=256 contact rollout has been started. Before it could support a spatial
claim, its input, exact binary, mesh-wise temporal refinement, contact-area
weights, physical outputs and reproducible traces must be checked against the
predeclared gates. Existing live references and the backward-Euler control
remain undisturbed.

## Extended parity and alternative reference evaluation

The frozen compact evaluator also reproduced the retained complete n=32,
h=0.5 microsecond, 10 ms trajectory. Its largest force difference was
5.51e-14 of the retained sampled peak; its cumulative impulse differed by at
most 2.37e-15 of the retained total. The mass and individual area weights
were exactly equal. The original retained trace, binary identities, limits,
observational elapsed time and resulting analysis are bound in
compact-reference-retained-n32-parity-001-analysis.json. An independent
reconstruction of each contact-area weight and a finite-trace check passed.
This is evaluator parity, not a new spatial or temporal convergence result.

A separate compile-time reference variant uses the six exact structured-cell
templates and axis spacings from the original rounded node coordinates to
avoid repeatedly inverting reference matrices. On independently deformed
n=2, 4 and 8 fixtures it agreed with the ordered evaluator to 5.47e-15 in
relative maximum acceleration, with identical mass and areas. Complete
10 ms traces at n=2 and n=8, h=2 microseconds agreed to at most 4.95e-13 of
sampled peak force and 1.23e-13 of cumulative impulse. Source and executable
hashes and the retained traces are in template-reference-trajectory-parity-001.json
and template-reference-deformed-parity-001.json. The template variant subsequently matched the retained complete n=32,
h=0.5 microsecond trajectory to 2.20e-12 of peak force and 2.74e-14 of
total cumulative impulse. The independently reconstructed area weights and
finite force traces passed for both n=32 reference variants; see the two
reference-n32-validation-001.json records and
template-reference-retained-n32-parity-001-analysis.json. Parity on n=32
still does not establish n=256 trajectory accuracy, temporal convergence,
spatial convergence or a reference-runtime benefit.
