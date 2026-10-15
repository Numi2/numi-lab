# Selector 5/6 outcome-gate audit (source 131)

This package preserves the failed native checker run `hybrid-runtime-armijo-outcomes-426` and a checker-only candidate patch. The patch fixes a diagnostic assertion that incorrectly treated a zero final relative-correction certificate as proof that Krylov work did not occur. It checks the explicit FGMRES iteration count instead. It does not change solver behavior, the applied force, material, time step, or any tolerance.

## What the retained run shows

The source-131 native run completed one microstep for selector 5 and selector 6. Each status is success with five FGMRES iterations; each acceptance certificate is valid and the nonlinear residual is `2.474832e-7`. Both report `nonlinear.y == 0`, which is the final relative-correction field. The two accepted FEM node snapshots and certificates are byte-identical. The original checker stopped first at the invalid assertion that demanded a nonzero final correction despite the recorded five Krylov iterations.

The unchanged subsequent loaded-node-motion gate is `> 1.0e-8 m`. Reconstructing node 2's initial FP32 position from the authored source-131 baseline deformation and reference coordinates, then comparing it with the raw final node snapshot, gives a displacement of `1.862645149230957e-9 m` for both selectors (one FP32 ULP at this coordinate). That is below the existing gate. Therefore correcting the telemetry assertion only exposes a separate fixture failure. The threshold is retained unchanged, and neither selector is qualified by this package.

The native run did not print its loaded-motion diagnostic because it exited at the earlier assertion. The included audit script derives motion from the raw captured state and the exact source-131 fixture literals. It binds the run receipt, stdout/stderr, prediction, source-build record, frozen-source manifest, and original checker hashes before reporting.

## Checker-only candidate

`checker-source131-assertion.patch` changes only the Krylov-work assertion in `checkSelector5And6AcceptanceParity`: it requires `statuses[0].fgmresIterations > 0`. The original motion assertion `loadedNodeMotion > 1.0e-8` remains unchanged. This candidate passed patch application and strict ObjC++ syntax checking. It was not run natively; its report is an audit of the retained source-131 outcome, not a new native pass.

## Reproduce the package checks

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/selector5-6-runtime-outcome-correction-assertion-001/validation/adjudicate_426.py
```

The report is written to `validation/fixture-audit.json`. `validation/patch-apply.log` confirms that the patch applied to the copied source-131 checker and produced the exact candidate file. `validation/strict-syntax.json` records the strict `xcrun clang++ -fsyntax-only` invocation. No GPU run was made for this package.

## Source and evidence

- Frozen source: `/Users/home/cardboard-implementation-20261014/snapshots/source-hybrid-outcome-capture-131`
- Native run: `/Users/home/cardboard-implementation-20261014/native/hybrid-runtime-armijo-outcomes-426`
- Frozen source manifest SHA-256: `8c076e2876df023b387f248b851d1224720e8096171f560ae72983c51d8e8b50`
- Original checker SHA-256: `1b91d8888c178cd5719be48eff8c1b0ee993d8078598ef6179e2dc45c44b1819`
- Native receipt exit code: `1` (retained checker failure); the solver reported successful accepted outcomes before the assertion.

No physical-assembly or preconditioner qualification is claimed. The minimal fixture's loaded-node motion remains below its unchanged requirement.
