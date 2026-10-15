# Cardboard coupon resolution study

`cardboard_resolution_study.py` seals a prediction and then runs a native Metal case list sequentially. It is a numerical sensitivity study for the admitted end-grip coupon; it does not simulate box assembly or establish physical accuracy. The plan is source-bound to the executable, metallib, relevant C++/Metal/header sources, material files, baseline run exports, and the exact `cardboard_convergence.py` reader. It refuses a changed input hash and does not overwrite an existing run root.

## Planned cases

The baseline must be a complete accepted export at `nx=8, ny=2, thickness_slices=1`, with positive loading, hold, unloading, relaxation, and free-release phases. The baseline phase counts, timestep, bend angle, material inputs, solver settings, and relative residual tolerance are frozen into the plan.

| Series | Cases | What changes |
| --- | --- | --- |
| Lengthwise mesh | `(8,2,1)`, `(16,2,1)`, `(32,2,1)` | `nx` only |
| Width mesh | `(8,2,1)`, `(8,4,1)`, `(8,8,1)` | `ny` only |
| Paper thickness mesh | `(8,2,1)`, `(8,2,2)`, `(8,2,4)` | through-layer slices only |
| Combined refined | `(8,2,1)` to `(32,8,4)` | direct interaction check; one pair, no three-level inference |
| Timestep | `(32,8,4)` at `dt`, `dt/2`, `dt/4` | timestep and all phase counts scale together |
| Slower loading, optional | `(32,8,4)` at base load duration and a declared slower duration | loading count only; separate `load_rate` comparison |

Spatial cases retain the baseline timestep and all five phase counts. In each temporal case, every phase count is multiplied by 2 or 4 while timestep is divided by the same factor. This preserves loading, hold, unloading, relaxation, and release durations and keeps the imposed loading speed fixed. The slower-loading case is a distinct comparison: it increases only loading duration and leaves hold, unloading, relaxation, release, timestep, and mesh unchanged. It is enabled with `--slower-load-factor N` and is never folded into the timestep series.

## Admission and gates

Before making metrics, the runner calls `cardboard_convergence._load_run` on every exported case and `cardboard_convergence.analyze_pair` on every comparison. Prefixes, rejected steps, incomplete certificates, mismatched phase schedules, and incompatible material or solver controls remain rejected/inconclusive. A failed or timed-out case stops later work and keeps its prediction, command, output, stdout, stderr, and status record.

For each three-level axis and timestep series, the measured difference from middle to fine must be strictly smaller than coarse to middle; an equal difference or a missing level does not establish decreasing error. The finest pair must satisfy all available gates:

- peak resultant reaction force: relative difference at most 2%;
- peak absolute reaction moment: relative difference at most 2%;
- absolute moment-path work: relative difference at most 2%;
- full common-time reaction-force curve: normalized RMS difference at most 5%;
- full common-time reaction-moment curve: normalized RMS difference at most 5%;
- change in measured free-release springback: at most 1 degree;
- paper damage-area and glue-debond-area fractions: absolute difference at most 0.02 when each mechanism is enabled.

Curve RMS is computed over all common accepted environment-0 times, normalized by the baseline RMS, with force components combined in newtons and moment treated separately in newton-metres. Temporal comparisons use the coarse accepted times shared with the finer run. Peak force and moment use every accepted sample in each run. Work is the integral of measured end-grip reaction moment against measured grip rotation; both signed external work and absolute path work are retained, and the 2% gate uses absolute path work. This is a numerical observable definition, not a claim that the path integral equals material dissipation.

Missing damage/debond fields are `inconclusive` when the corresponding mechanism is declared enabled. Feature enablement must match across a pair. No stage-resolution claim is made unless every run carries explicit native certificates for crease-tool contact, paper damage, and glue debond. The current end-grip coupon lacks these certificates and area observables, so even a successful mesh/timestep screen is labeled `numerical_screen_only_stage_scope_incomplete`. The current convergence loader also excludes the tool-driven crease mode, so this tool does not admit or qualify scoring-contact runs.

The solver's relative residual setting is held constant across spatial, temporal, and slower-loading cases. Because the solver does not export the normalization scale behind that relative residual, the plan does not infer equal absolute force accuracy from the tolerance or from `tolerance/dt`. Every accepted run's native free-force imbalance is reported in newtons for both environments. An optional `--max-free-force-residual-n VALUE` predeclares a bound on the observed L2 and max-component residuals; without that declared bound, the group-level force-residual gate stays inconclusive. This is a bound on the exported residual, not an estimate of force error.

## Running

Plan without launching Metal work:

```sh
python3 matter/tools/cardboard_resolution_study.py plan \
  --baseline /path/to/complete-8x2x1-release-run \
  --probe /path/to/numi-matter-cardboard-probe \
  --metallib /path/to/retained-cardboard.metallib \
  --run-root /path/to/new-resolution-study \
  --timeout 1800 \
  --slower-load-factor 2
```

If the project has a declared absolute residual cutoff, add `--max-free-force-residual-n VALUE` before planning. The tool provides no default or literature-derived cutoff; without one, it retains an inconclusive force-residual gate.

Run the sealed plan on the configured macOS Metal host:

```sh
python3 matter/tools/cardboard_resolution_study.py run \
  --plan /path/to/new-resolution-study/study-plan.json
```

The runner uses a host-wide `flock` file and launches one case at a time. It does not resume into existing case directories or replace failed evidence; make a new plan/root for a retry. No native run is launched by the CPU contract tests.

## Verification

```sh
python3 matter/tools/test_cardboard_resolution_study.py
```

These tests use synthetic labeled fixtures to exercise mesh scheduling, physical-duration preservation, analyzer rejection, threshold arithmetic, and missing damage-field behavior. They are not solver, material, mesh-convergence, or physical-validation evidence.
