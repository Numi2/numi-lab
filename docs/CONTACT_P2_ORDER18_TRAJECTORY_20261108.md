# Independent P2 order-18 dynamic trajectory

The retained-state study supports order 18 on the earlier trajectory; it
does not prove accuracy after forces and motion change. This preregistered
CPU experiment repeats the complete P2 n=2 evolution with order-18 surface
quadrature, the same regularized continuum energy, consistent mass and all
prior physical parameters. This is a new spatial/contact discretization,
not the production nodal P1 law.

## Prediction and gates registered before execution

Compare 10 ms coarse and fine DOP853 runs at max steps 8 and 4 µs,
rtol 1e-8 and 1e-10, displacement atol 1e-12 and 1e-13 m and velocity
atol 1000 times those values. Force bins, integrated impulse and contact
work must agree within the unchanged 0.25% limits with the same normalization
as the earlier P2 study. At every accepted output snapshot evaluate the
force and gaps using independent 48×48 and 64×64 positive Gauss-Legendre
Duffy rules on each surface triangle. Their force difference must be <=0.05%
of the fine peak 1 ms force bin; order-18 versus Duffy-64 force difference
must be <=0.25% of that bin. Predict both will pass, while retaining all
failure if they do not.

At every ODE evaluation reject nonpositive order-18 quadrature gap or volume
Jacobian. Require positive sampled Duffy gaps, momentum imbalance <=0.25%
of fine total impulse, total-energy drift <=0.25% of the free-fall kinetic
scale, and contact work plus potential change <=0.25% of maximum sampled
contact potential or absolute work (floor 1e-12 J). These are diagnostics,
not a proof of geometric clearance between quadrature points or time
samples. Preserve all states, inputs and results; use single-thread BLAS,
no GPU, sequential 360/720-second process-group timeouts, and stop on
failure.

Even if this pair passes, n=2 is not spatially qualified. No production
contact, replay, hard-contact or physical-material claim follows.
