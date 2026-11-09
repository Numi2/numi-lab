# Independent stage-correction feasibility check

The retained trace shows all vertical corrections at two stalled iterations
rounding away when added to the stored FP32 stage. Before changing the runtime,
evaluate the retained iteration 3 candidate and proposed correction in the
independent FP64 fixture equations. Construct full endpoints from the accepted
high/low state and either the captured stage or the stage plus the exact FP32
proposal interpreted in FP64. Represent the constructed endpoint velocity and
reference displacement as high/low pairs. Keep the physical law, geometry,
step, mass and acceptance limit identical. This is a noncontact, offline
single-correction diagnostic, not a runtime solution, a proof of convergence,
or contact qualification. Record residual and kinematic defects for both
states, and preserve the result even if the proposed correction worsens it.
