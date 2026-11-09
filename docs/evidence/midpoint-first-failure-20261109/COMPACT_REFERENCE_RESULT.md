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
