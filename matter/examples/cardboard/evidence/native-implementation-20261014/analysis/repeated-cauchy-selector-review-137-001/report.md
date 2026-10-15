# Repeated Cauchy selector overlay review

Review is read-only and bound to candidate `matter/src/runtime.mm` SHA-256 `827eeecfa8c85b9c4d955da4b0fced2e9c479d0da0a9f8c419d9126e1049651e`, against frozen source136 `runtime.mm` SHA-256 `8c56679c1e547da78f5973382a07ea371b0f3492bebd611cfe73e15f952c959b`. Frozen snapshot manifest SHA-256: `3d76d2cdec4f0e58ac9de2fc158115b386049e019204ae7a466da956a9a81dae`.

## Findings

The selector parser is fail-closed. It accepts a decimal control step followed by exactly `:all` or a decimal Newton iteration in `[0,256]`; empty tokens, signs/whitespace, extra suffix text, missing `:`, root overflow, and out-of-range Newton values fall through to rejection. The null-separator path is guarded before `separator + 1` is evaluated.

For `control:all`, `baseCauchyNewtonSelected` is true on every nonlinear Newton loop iteration while `femBaseCauchyFallbackSelectedRoot` restricts execution to the selected control step. The Cauchy direction remains the existing second Armijo pass. This does not alter the contact/operator assembly, candidate preparation, acceptance, commit, or rollback code.

At the selected root, the mode requires `NM_FEM_TRUE_RESIDUAL_CERTIFY=1`, a matching true-residual root, and rejects any `NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR`. In certification mode `femTrueResidualEnabled` and replacement are enabled each FGMRES restart cycle; the existing true-residual measure recomputes `b-Ax` and updates the convergence latch. Thus a filtered true-residual selector cannot qualify `:all`.

The default path and exact `root:iteration` selector retain their prior selection condition and 64-record trace cap. Changes are limited to parser state, selection of all Newton iterations, and a conditional increase of the residual-Armijo trace/base-status cap. The observer cap does not control whether the fallback runs.

One correction is required for accurate evidence labels before freezing: the trace-enable event at candidate line 11469 still reports `maximum_newton_records:64`, and the cap event at line 11797 still reports `record_cap_64`. In all mode the actual cap is 512. These labels do not alter solver behavior, but make the resulting trace receipt contradict the configured observer ceiling. The 512-record observer is bounded; because the runtime permits a caller-supplied Newton budget above 256, it can truncate observation while the numerical fallback continues. The existing stop event should report the actual selected cap, and the enable event should either be updated after selector parsing or explicitly report the default cap plus a later all-mode event.

## Validation

Strict Objective-C++ syntax-only compilation passed with `-Wall -Wextra -Wpedantic -Werror` and frozen136 headers. No Metal compilation, native execution, GPU run, or physical step was performed.

Command:

```sh
xcrun -sdk macosx clang++ -x objective-c++ -std=c++20 -fobjc-arc -fsyntax-only -Wall -Wextra -Wpedantic -Werror \
  -I /Users/home/cardboard-implementation-20261014/snapshots/source-cache-projection-reuse-runtime-136/matter/include \
  -I /Users/home/cardboard-implementation-20261014/snapshots/source-cache-projection-reuse-runtime-136/matter/src \
  -I /Users/home/cardboard-implementation-20261014/snapshots/source-cache-projection-reuse-runtime-136/include \
  -DNUMI_MATTER_DEFAULT_METALLIB=\"\" \
  /Users/home/cardboard-implementation-20261014/analysis/repeated-cauchy-selector-overlay-137-001/candidate/runtime.mm
```

Verdict: the numerical selector and same-state true-residual restrictions are internally consistent by source inspection. Treat the candidate as review-pending only for the stale trace-cap labels; this review is not a native qualification.
