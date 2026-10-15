# Source380 tet674 material trace diagnostic (v1)

This package adds a bounded, single-point diagnostic to the existing source112 `numi-matter-cardboard-bilinear15-capture-diagnostic-check` target. It reads the exact source380 compiled Matter package, binds the medium source asset and run artifacts by SHA-256, verifies the accepted state against the rolled-back history export, and dispatches the existing production material stress/tangent path plus a trace-on/off comparison for environment 0, tetrahedron 674.

It is an observation-only tool. A completed dispatch is not a constitutive pass, solver pass, coupon qualification, or physical validation. The output records projection failure/iteration/status without asserting the expected outcome. The source380 F high/low words are still the earlier CPU transcription bound to the exact candidate/package hashes; `F_words_native_readback` is explicitly false. Root is preparing a separate native deformation-word capture using the production deformation kernel. Do not treat this diagnostic input as native confirmation of F.

The trace capacity is 32 local Newton iterations and 16 state slots, sufficient for the exact compiled medium material (15 states, local budget 32). Only the diagnostic trace arena capacity is enlarged; constitutive equations, thresholds, material parameters, history, and production world are unchanged. The trace and trace-off outputs are compared bitwise for stress, tangent, and projected high/low state; mismatches are reported rather than converted into an expected pass/fail.

The source380 run failed before global FGMRES/outer Armijo, at control step 1, environment 0, Newton 0, tet 674, with status 5 (`NM_STATUS_INVALID_DEFORMATION`). This checker does not decide whether the local projector caused that status.

## CPU input preflight

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/source380-generalized-material-point-001/verify_input.py \
  /Users/home/cardboard-implementation-20261014/analysis/source380-generalized-material-point-001/native_point_input_674_v2.json \
  /Users/home/cardboard-implementation-20261014/snapshots/source-explicit-material-budget-109/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter
```

This verifies the point input's packed FP32 words, candidate-node/history/mesh/package hashes, and accepted-state words. It launches no native or GPU work.

## Native command

After integrating this overlay into the exact source112 base and building the existing diagnostic target, run:

```sh
numi-matter-cardboard-bilinear15-capture-diagnostic-check \
  --native-source380-point \
  --expected-input-sha256 7925a6eba993e6bf8b4f536352027d18c221f95c215fe73d1b20ebaec6aeb797 \
  --expected-package-sha256 cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c \
  --expected-world-fingerprint 8485105279886344921 \
  /Users/home/cardboard-implementation-20261014/analysis/source380-generalized-material-point-001/native_point_input_674_v2.json \
  /Users/home/cardboard-implementation-20261014/solver/explicit32-certified-coupon-380/output/compiled.nmatterpack \
  /Users/home/cardboard-implementation-20261014/snapshots/source-explicit-material-budget-109/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter \
  /Users/home/cardboard-implementation-20261014/solver/explicit32-certified-coupon-380/output/rejected-conditioning-000002/candidate_nodes.bin \
  /Users/home/cardboard-implementation-20261014/solver/explicit32-certified-coupon-380/output/material_state_000002_bent.json \
  /Users/home/cardboard-implementation-20261014/solver/explicit32-certified-coupon-380/output/mesh.json \
  /Users/home/cardboard-implementation-20261014/analysis/source380-generalized-material-point-001/native-output
```

The command reads and validates all named source artifacts before dispatch. It does not take physical simulation steps or modify source380 artifacts. This package has not been C++/Metal compiled or run on GPU; those are root-owned follow-up checks.

## Erratum carried forward

`erratum_source380_tet674.json` corrects two stale descriptive values in the prior CPU audit without modifying it: the compiled temperature is `bulk.y = 293.1499938964844 K`, not 0 K, and the source-transcribed candidate determinant delta is `-1.3981052804368233e-7`, not `0.002556720636401888`. The original audit and its hashes remain preserved.
