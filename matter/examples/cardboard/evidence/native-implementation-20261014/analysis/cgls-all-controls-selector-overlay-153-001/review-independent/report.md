# Independent review: all-controls CGLS selectors (source153)

## Result

The parser change is narrowly scoped and preserves numeric selector behavior. Exact `NM_FEM_TRUE_RESIDUAL_ROOT=all` resolves to the current control step, then uses the existing capability, certification, optional Newton selector, and restart checks at that control. Exact `NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=all:all` resolves to the current control and all Newton iterations, but is admitted only when bounded CGLS iterations and the CGLS primary policy are configured. The existing selected pure-FEM residual-Armijo capability checks, true-residual certification requirement, static FEM caps, positive finite mass check, scope check, per-environment eligibility, Armijo acceptance, final certificate, and rollback remain downstream. The selector patch does not change solver thresholds or operator actions.

The extracted-source CPU contract harness reports `selector_checks=44 passed`; the source153-bound build receipt reports `numi-matter-cardboard-probe` built successfully. These establish parser/compiler evidence only. No native solver run or physical/convergence claim is included.

## Operational constraint: singleton capture paths

Do not set `NM_FEM_BASE_CGLS_CAPTURE_PATH` or `NM_FEM_BASE_CGLS_STATE_CAPTURE=1` for an all-controls run unless the implementation first gains a per-control output namespace. The CGLS scratch snapshot is selected by Newton index, independent of control step (`runtime.mm` lines 22996–23005), and writes to the same path with `O_EXCL` (22963–23069). The state capture path and its `.state`/`.regional` companions also lack a control-step component and use exclusive file creation (source151 runtime lines 22897–22904, 23311–23338, 23571–23575). With `all:all`, the same Newton index can be selected at multiple control steps: the first write occupies the path and later writes cannot create a second record there. This is a diagnostic completeness/collision issue, not a solver acceptance change. True-residual cycles are emitted as stderr JSON lines and do not use a singleton output file.

## Source checks

- `NM_FEM_TRUE_RESIDUAL_ROOT=all`: exact-string branch at source153 `matter/src/runtime.mm:11831–11855`; the current `request.controlStep` is selected, then `femTrueResidualCapabilityFailure()` and optional selector checks execute before enabling.
- Cauchy `all:all`: exact `all` root and `all` Newton suffix parsing at `12018–12048`; the new all-control gate requires nonzero bounded CGLS and primary-policy request. After root resolution, existing per-control admission continues at `12060–12104`, and global primary-policy consistency is checked at `12137–12140`.
- Numeric selector path remains the prior decimal parser and range checks; malformed near-matches (`All`, whitespace, extra delimiters, overflow, invalid suffix) reject in the 44-case extracted-source harness.
- The source153 frozen manifest records 787 source files and 1,034 configure-support files, with runtime and frozen manifest as the only changed files from source149.

## Boundaries

This review does not establish all-control capability admission for every future control state; the runtime repeats its existing checks and may reject a later step. It does not establish that a complete physical coupon converges, nor that an all-controls run is numerically beneficial. The singleton capture limitation above must be handled in the launch protocol.
