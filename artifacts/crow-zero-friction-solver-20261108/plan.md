# Exact zero-friction native solver correction

The frozen grasp control rejected at tick 0, stage 7, because the unified quality
contact assembly requires positive cone scales. A zero-friction isotropic
contact is the normal impulse interval with tangential impulses fixed to zero.
Assemble that contact as an existing scalar interval; retain its normal
regularization, adhesion, impulse cap and solver quality gates. Initialize
unowned impulse rows on every objective evaluation. A partially zero or
otherwise invalid anisotropic cone stays rejected.

Validation, in order after the original open-jaw control: build native shader;
run existing unified solver and world-contact probes; run a bounded native
frictionless arm with the original crow fixture, original dylib and the new
shader; inspect all 640 transitions, physical bounds and exactly zero tangent
impulses; run the normal arm with the same corrected shader and compare its
physical files with the original normal control. Retain unsuccessful attempts.
These controls concern an estimated sphere fixture, not food intake or learned
grasping.
