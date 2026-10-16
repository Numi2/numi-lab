# Source246 full-box force residual and stage certificate policy

This candidate starts from frozen source241 and composes the sealed source242 constituent-volume observables patch. The source246 delta adds accepted full-box free-node residual-in-Newton measurements and an optional explicit residual bound, and makes the stage analyzer honor the native certificate semantics without treating correction telemetry as an additional convergence gate. Time refinement checks both compatible relative-residual-tolerance/dt controls and a separately measured residual bound in N; the ratio alone is not presented as an absolute force bound.

The force measurement is the final accepted FEM residual xyz divided by the accepted timestep at nodes with restAndFixed.w == 0. It is an observed solver residual, not an estimate of physical force error. The CLI bound has no default and does not alter the solve. Historical manifests without the policy preserve their historical correction gate.

Validation is CPU/source-bound only: probe and helper targets compile, helper/admission/stage-study tests pass. No native full-box physical run or GPU qualification was performed.
