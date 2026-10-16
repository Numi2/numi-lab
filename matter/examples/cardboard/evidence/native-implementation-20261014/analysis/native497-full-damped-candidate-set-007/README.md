# Native 497 full-total-step candidate set

This package contains CPU-only candidate vectors reconstructed from the same captured FP32 tangent-cache operator for native 497, environment 0, free XYZ rows only. It contains no native dispatch or nonlinear acceptance result.

The iteration-limited LSQR report and lambda=0 array retain the requested candidate: it is the complete total step from zero, stopped at the 5,000-iteration cap with status 7, norm 0.032622929050346576 m/s, and true residual 4.245294865562062e-6 Ns (0.22525974978091182 of ||b||). Do not present it as a converged solve.

The separate dense binary64 SVD report records solutions to min ||A x-b||^2 + lambda^2 ||x||^2, also from zero. Its lambda=0 vector is a full-step solve of the captured matrix, distinct from the capped LSQR result. Positive-lambda candidates are damped total steps; none inherits the native Newton certificate.

The two source reports and all candidate arrays are copied byte-for-byte. Source operator/report lineage is recorded in the JSON reports. The capture is one mechanical free-XYZ system only; A is reconstructed in binary64 from the captured FP32 tangent cache, not an independent constitutive reference. Nonlinear reference replay is required to assess direction usefulness.
