# Source064 angular free-BOX overlay audit

Read-only review of `/Users/home/cardboard-implementation-20261014/analysis/angular-box-contact-063-overlay` against frozen source063 commit `5fd4486ea2bcbe903782509a769ce3935b790e55`. The source overlay manifest SHA-256 is `0a2728eca70bf37751ef33973623cb3521f9ce47e4084b33e6124699f2a15c1e`; patch SHA-256 is `1e2ef15649f004a56e79dfe9c4ce1bb1f75a5e6da87209e2cd25b768c3c967a7`. Exact source, build-artifact, and receipt hashes are recorded in `manifest.json`.

The overlay and frozen-base file bindings match, and `git apply --check` exits 0 on the frozen 063 source. The independent CPU formula suite passes 11/11. Metal compilation, metallib generation, and Objective-C++ syntax checking each passed (exit 0). No native GPU run was performed.

I found no actionable correctness issue in the final overlay. The tested angular path differentiates the same left-multiplied SO(3) pose used by the candidate, includes witness and normal-frame motion, inverse-response and rotating world-inertia terms, and includes the proxy-center offset in the conservative swept radius. The spatial normal differential is scaled by `dt`, consistent with the displacement-space differential and CPU finite-difference checks; the geometric contribution is not double-counted.

The implementation remains bounded to one non-articulated six-DOF free BOX, smooth frictionless features, and its certified alpha/full-timestep paths. Frictional, nonsmooth, impact, multi-body/articulated, adaptive-rate, MPM, and mixed/cohesive cases remain outside the verified scope. Historical native cases 195/196 predate the overlay and fail the old fixed-orientation gate; they do not qualify this package.
