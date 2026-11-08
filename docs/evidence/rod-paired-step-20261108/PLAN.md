# Research-only paired-position axial step

The prior finite-lattice result and Metal arithmetic test are retained at
`docs/evidence/rod-fp32-representability-20261108/`. Test whether carrying
paired FP32 coordinates, attachment targets and axial parameters through the
existing native standalone rod entrypoint and real Metal step, with paired
readback to its double state, can satisfy the previously failing local momentum
and position/velocity/reaction gates. This is an isolated research build; do
not modify production entrypoints or enable CCD/AL experimental options.

Admit only straight positive-x chains with exactly aligned velocities and hard
anchors, no tool/rigid/self contact, no gravity, damping, twist or precurvature.
Unsupported input must remain on the previous guarded solver, and a failure to
certify must reject transactionally. Other axes, bending and coupled contact
are deliberately outside this first test and remain required for completion.

Replicate the 24 earlier five/eight-node x/oblique, fixed/moving/internal,
0/0.25 m cases. Oblique controls must not be accepted by the x-only candidate
as if their transverse components were zero. Compare accepted x cases against
the retained independent FP64 reference and full energy finite-difference
momentum calculation: node position <=1 µm, velocity <=0.001 m/s, force error
<=max(1 µN,1%), work error <=1e-8 J, and local momentum <=0.001 m/s. Preserve
all previous failed and accepted results. Run each twice; require identical
physical outputs and report rejected/failed trials, not just passing cases.
If supported cases fail, retain and diagnose; do not relax gates. Fresh
six-node cases at offsets 0.4, 1.25 and 100 m with fixed/moving/internal
anchors are held out for a second predeclared test only if replication passes.

The experiment includes only one-step no-contact synthetic mechanics. It does
not qualify persistent MetalWorld state, finite twists, arbitrary orientations,
reaction coupling, full restoration, multiple steps, surgical motion or speed.
No production promotion until those are separately implemented and verified.
Record source/patch/binary/metallib/input hashes, commands, and raw outputs.
