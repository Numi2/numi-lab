# Full-stage shipping-box resolution analysis

`cardboard_stage_resolution_study.py` is a read-only adapter for retained native
FEFCO 0201 exports. It does not start a simulation, convert a failed prefix
into a baseline, or publish physical qualification. The existing
`cardboard_resolution_study.py` remains the executable study runner for the
smaller end-grip coupon; this adapter adds strict admission and comparison
planning for the shipping score, assembly, compression, gravity-transfer, and
drop ledgers.

## Admission contract

Every shipping run must contain `manifest.json`, `result.json`,
`box-native-result.json`, `box-native-steps.csv`, `box-actions.csv`, and
`box-score-angles.csv`. The per-step CSV must provide accepted state, native
status, certificate validity, residual, correction, volume and pressure
residuals, simulation time and timestep. Each row must be accepted with zero
native status and each exported certificate value must satisfy the matching
manifest tolerance. Row counts and source/world fingerprints must agree with
the native result. Old exports without the new certificate columns remain
inconclusive; preflight-only results are explicitly not physics baselines;
rejected steps, failed statuses, incomplete stage branches and truncated
histories remain failed prefixes.

The adapter also reopens the liner, medium, and glue source files named by the
manifest and verifies their SHA-256 digests. If an enabled retained-matrix
damage, starch-interface, or HMA law declares a parameter file, that file is
also reopened and verified. Missing or changed source files prevent a complete
candidate from becoming a comparison baseline. Failed runs stay classified as
failed prefixes even when their older step exports lack the newer certificate
columns.

The adapter separately checks the requested scope. Scoring requires all 12
score lines and withdrawn tooling. Assembly and downstream scopes require a
settled, released assembly and a zero-active-clamp gate. Compression requires
complete nominal and half-speed branches plus the native rate-sensitivity
gate. Impact additionally requires all declared gravity-transfer and drop
cases. An accepted native run can still be unqualified: analogue laws and
engineering geometry remain distinct from source-grade calibration and
physical-box validation.

Pair comparisons require matching geometry, constituent source hashes,
damage/cohesive parameter identities, contact settings, solver controls,
tools, and process stages except for the declared axis. Mesh comparisons may
change one declared mesh coordinate only. Timestep comparisons additionally
require identical per-stage accumulated physical duration and prove actual
refinement from accepted `box-native-steps.csv` rows. For both nominal and
half-speed compression branches, each `shipping-compression.csv` observation
must bind to its consecutive native substeps; its control-step count, summed
duration, minimum dt, and maximum dt must match those rows. The timestep pair
must preserve each branch duration while the finer run has a strictly smaller
maximum accepted dt and a larger accepted substep count. Manifest labels alone
do not establish timestep refinement. Score-depth
comparisons are separately labeled sensitivity runs and require a fresh flat
blank; they are never treated as convergence levels. Compression nominal vs
half-speed is a within-run rate-sensitivity comparison, not timestep
refinement.

## Available and unavailable observables

When complete, `shipping-compression.csv` supports a platen force–stroke
curve and a separately integrated force–stroke work comparison. The adapter
retains the native cumulative work values as a cross-check. Force is reported
in raw N; the curve-RMS difference is normalized by the left-run RMS force
and is not divided by board width. Compression timestep admission also
checks each macro observation's `macro_observation_duration_s`,
`control_step_count`, `control_step_dt_min_s`, and
`control_step_dt_max_s` against the accepted native substeps.

`box-grip-reactions.csv` supplies accepted clamp/fixture reaction moments.
The adapter groups them by stage, action, score line, and environment and
compares matching active-grip series. Moments remain raw N·m, not N·m per
width; the curve difference is normalized by the left-run RMS moment. This is
the clamp/fixture wrench moment, not a score-punch contact moment.
`box-score-springback.csv` supplies held-to-released score-line turn changes.
The adapter admits those rows only when native release/settle gates pass,
every line and environment is present, endpoints bind to accepted native
steps and times, and both shape-fit residual and coverage gates pass. The
probe's engineering shape-fit bound is explicitly not a published
manufacturing tolerance. The score-angle CSV remains action-relative and is
not substituted for released springback.

The accepted-history CSV exports area-weighted constitutive paper and cohesive
degradation indices; it does not export geometric paper crack area or a
thresholded open/debond area. Those indices are reported separately and do
not satisfy the missing geometric fractions. Therefore the full-stage gate
remains inconclusive until those native observables are available, even if
force/work comparisons can be computed.

For mixed-mode cohesive faces, the newer history export also reports the
reference-area integral of fully failed accepted quadrature points, divided by
the total activated reference area for that interface group. Each mixed-mode
face has three equal reference-area weights. The maximum damage stored in
face.state.x is not an area fraction: one failed point contributes one third
of that face area, and a fully failed face count requires all three points.
Legacy exports using the face maximum are retained but cannot satisfy this
capacity-area comparison. The numerator means zero remaining tensile and shear
capacity at those quadrature points in the constitutive law. It
does not mean a visible gap, a geometrically open face, or observed physical
debonding. Starch and HMA are accounted separately; dormant faces are not in
the denominator. Reports retain the partial area-weighted degradation index
as a different metric.

## Commands

Write an admission report or a plan without launching native work:

```sh
python3 matter/tools/cardboard_stage_resolution_study.py inspect \
  --run /path/to/native-output --output /path/to/new-admission.json

python3 matter/tools/cardboard_stage_resolution_study.py plan \
  --reference /path/to/native-output --output /path/to/new-stage-plan.json
```

Compare two fully admitted runs or a three-level family. A triplet only reports
whether available pair differences decrease; it will not claim full-stage
convergence while required observables are missing.

```sh
python3 matter/tools/cardboard_stage_resolution_study.py compare \
  --left /path/to/coarse --right /path/to/fine --axis mesh_nx \
  --scope all --output /path/to/new-pair.json

python3 matter/tools/cardboard_stage_resolution_study.py triplet \
  --coarse /path/to/coarse --middle /path/to/middle --fine /path/to/fine \
  --axis time_step --scope all --output /path/to/new-triplet.json

python3 matter/tools/cardboard_stage_resolution_study.py compression-rate \
  --run /path/to/complete-compression-run --output /path/to/new-rate.json
```

Triplets reject score-depth and score-speed sensitivity families as
convergence axes. Spatial and time triplets require strictly ordered levels;
impact triplets accept a matching list of timestep levels. The separate
`capacity-qualification` producer uses one shared middle run across the mesh
and timestep triplets, stores both adjacent metric vectors, and rejects a
triplet if any peak/curve/work difference increases from coarse-to-middle to
middle-to-fine (at least one must strictly decrease). When the selected scope
includes scoring or assembly, accepted grip moments and released springback
are included in the pair vectors; missing or failed observables stay
unresolved. The report always keeps `convergence_claim: false` because
geometric paper-crack and starch/HMA debond areas remain unavailable. Scope
`compression` intentionally does not claim or compare score-line moment or
springback observables.

Reports are source-bound to SHA-256 digests of every present required and
optional run export. Output creation refuses to overwrite a prior report.
Synthetic contract tests are run with:

```sh
python3 matter/tools/test_cardboard_stage_resolution_study.py
```

These fixtures validate the analyzer's gates only; they are not native
mechanics evidence.

## Irreversible debond fraction and capacity admission

A fully failed cohesive quadrature point has permanently lost tensile and shear
capacity. Its reference area counts as constitutively debonded even if subsequent
closure brings the faces back into compressive contact. The 0.02 absolute fraction
gate uses the three equal reference quadrature weights, accepted D exactly 1, and
the same activated reference-area denominator on each mesh. It reports starch
and HMA separately. Geometric opening remains a separate unavailable observable.
Legacy maximum-face-damage exports and incompatible activation coverage are not
admitted.

Capacity qualification now requires accepted paper-damage and irreversible-debond
comparisons as well as decreasing force, curve and work differences. A force-only
match cannot qualify Fpeak while an enabled paper localization field remains
unmeasured. The current per-tet area-weighted damage index is retained but is not
substituted for a mesh-objective fractured-area fraction. Its missing observable
keeps the damage-enabled capacity gate inconclusive. Explicitly disabled mechanisms
may be not applicable; that does not qualify a damage-enabled cardboard model.
