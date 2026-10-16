# Independent source153 review

This review checks the bounded all-control selector overlay against its
source149 parent, extracted-source selector checks, and source-bound build
receipt. It is a CPU/source review only.

The overlay changes only `matter/src/runtime.mm` in the source tree. Exact
`NM_FEM_TRUE_RESIDUAL_ROOT=all` resolves to the current `request.controlStep`
and still runs the original true-residual capability/certification checks.
Exact `NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=all:all` resolves that root to
each control step and all Newton iterations. It rejects the all-control form
unless the bounded 32/64-iteration CGLS path and the CGLS-primary policy are
requested. The later admission still requires same-control true-residual
certification and the existing pure-FEM, positive finite masses, cached
tangent, and scope checks. At each selected step the direction is installed
through the existing Armijo and final-certificate path; no physics operator,
material update, acceptance threshold, or rollback rule changes.

The source-extracted C++ parser harness passes 44 checks, including malformed
selector cases and numeric selectors. The source153 build receipt passes for
`numi-matter-cardboard-probe`; neither check runs a native solver.

The path-based CGLS binary capture and `.state` capture use exclusive file
creation and do not include a control-step component in their path. With
`all:all`, a selected Newton can recur at later control steps and collide with
the earlier file. Omit these singleton capture options or add
control-step-specific namespacing before collecting all-control snapshots.
`NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT` remains a numeric single-control selector.
The all-Newton tracing branch chooses a 512-record cap, while the generic
enable notice describes the default 64-record budget. The notice is default
metadata and should not be read as the selected cap; neither budget guarantees
complete trial traces across an all-control coupon.

This review establishes selector/source/build behavior only. It does not
establish native default parity, successful convergence, complete coupon
execution, or physical qualification.
