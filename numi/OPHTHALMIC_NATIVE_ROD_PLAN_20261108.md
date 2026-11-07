# Effective 8-0 Nylon axial rod: native CPU diagnostic

## Prediction

Can the existing native discrete elastic rod, using an effective linear relation inferred from a single physical specimen, reproduce that relation over the measured 2–4 mm extension interval with consistent reaction and work? Predict that all accepted samples agree with the affine relation to 1% relative force and that accumulated target work agrees with the change in elastic plus kinetic energy to 1% relative, without rejected steps. Separately compare the native force at each held-out data sample's extension with the original four-specimen 10% relative absolute-error criterion. Report each condition even on failure.

## Fixture fixed before execution

Use the existing `makeStraightSutureRod` and `stepDiscreteElasticRodCpu` implementation at a recorded source commit, compiled on ssh numi. Two nodes, zero gravity, no collision or damping, 1 ms timestep, 4 seconds. Both endpoints are hard-attached; the left is stationary and the right moves in the rod axis at 0.5 mm/s, from 2 mm to 4 mm measured extension relative to the paper's 120 mm test gauge. Initial geometry starts at the 2 mm mark with a linearly interpolated endpoint velocity; this is an explicitly constructed effective reference state, not a restored surgical state.

The fitted law is `F=a+b*x`, with `a=0.04886680223398625 N` and `b=0.020256155067237984 N/mm`. Choose effective rest length `L0 = 0.120 m - a/(1000*b) m` and axial stiffness `EA = 1000*b*L0 N`. This reproduces that local affine force law while making the offset explicit; it does not establish the specimen's actual stress-free length or identify Young's modulus or cross-sectional area. Radius and density are numerical fixture inputs, not calibrated material data. Do not adjust them after seeing results.

Reaction force is the signed rod force on the moving target; external actuator force is its negative. Work uses trapezoidal external force integrated against actual target displacement, with starting and ending elastic and kinetic energies checked independently. Use the original predeclared held-out data, unmodified; linear interpolation between adjacent native output samples is allowed to compare the measured extension coordinates and no time shifts or smoothing are permitted.

Any rejected step, incorrect reaction, unexplained energy/work mismatch or held-out error over 10% fails its own gate. Preserve failing data and do not relax the criteria. A passing result would cover only an effective one-dimensional CPU rod segment and this interval: it would not qualify Metal, bending, twist, rate effects, friction, fracture, tissue or actual suturing. Native Metal execution and state restoration remain independent gates.
