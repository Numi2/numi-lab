# Native 041b default32 continuation — failed

This run requested 32 steps at 125 µs (4 ms) and exited 1 after 21 accepted roots (2.625 ms), with wall time 355.814100708 s. It failed at root 21, tick 3: Matter status 6, object 19, local failing index 2623, three microsteps, diagnostics `[3.7129670005242588e-08, -4.6978958707412106e-15, 1, 0]`. The exact invocation, provenance, execution record, common failure receipt, full 21-row coupled CSV, surface audit, and 9.6 KB stderr are retained. Stdout and large output artifacts were not copied; original paths, sizes, and SHA-256 values are recorded in `raw-log-identities.json`.

The run used the same tested 041b source map and executable/metallib/library hashes as `verification-041b`. The retained source patch is `../tested-candidate-041b.patch` (SHA-256 `baf519eedf284db531dffc99c9ad460924880843d98ed174432203f822964625`); no source files or patches were changed while packaging.

The predeclared prediction in `TRIAL.md`—32 accepted steps with no rejected publication—is refuted: this run stopped at 21 roots, four roots earlier than the referenced 039 control's 25-root frontier. This result gives no stability improvement. It is not a sustained-support or physiology result. The earlier 041 preparation failure and the short passing 041b verification remain preserved as separate records.
