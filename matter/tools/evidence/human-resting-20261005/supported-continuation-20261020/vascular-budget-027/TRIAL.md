# Bounded vascular linear-budget trial 027

Diagnostic 026 exhausted its 32-column mixed FGMRES budget at root 20 /
microtick 3 / Newton iteration 1. Its final Arnoldi residual estimate was
0.0006592377904, above the unchanged 1e-5 vascular bound; convergence was
false. The original vascular merit gate rejected the available best direction.

Use the existing runtime iteration-budget API with 64 columns, restart width
16. Default behavior remains unchanged. No equation, force, physical timestep,
material, geometry, certificate or positivity policy changes.

First build and run the unified rollback regression. Then request 32 x125us
in the identical full scene with the 64-column override and existing root20
vascular trace. Check the original failure point, each accepted physical row,
unchanged mass accounting and final transaction status. A different accepted
trajectory is permitted because earlier exhausted linear solves can improve.
A completed short horizon is not supported-pose or long-run qualification.

If this bounded increase fails to improve the failing direction, retain the
failure and change the solver method rather than repeat larger budgets.
