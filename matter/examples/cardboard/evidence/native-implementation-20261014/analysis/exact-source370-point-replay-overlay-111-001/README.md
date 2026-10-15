# Exact source-370 material-point replay (overlay 111-001)

This diagnostic isolates the first rejected material point from source-bound run 370: environment 0, tetrahedron 1820, medium material 1. It feeds the exact rejected `candidate_nodes.bin` and accepted rollback material-state row through the production FEM deformation/material-frame helpers, then evaluates the actual production stress and tangent/projector entry points at private source-only compiles with local update budgets 16 and 32.

The retained run requested 32 local updates, but its compiled material descriptor is 16. The checker verifies and reports that mismatch. Its 32-update result is a one-point diagnostic recompile from the same source material and parameter words; it is not a replay of the full 370 trajectory and does not rewrite the run package or receipt.

## Source and input bindings

The patch parent is frozen source 108 (`snapshots/source-selected-capture-tail-reuse-108/frozen-source.json`, SHA-256 `251a8294ef2c4e7517ab7d8497c5f30d6ccc2d9523e7cb383f96f9ae220c6491`, parent revision `1648fe0e0483b5b05ce3c1e64d78cbec4885899e`). The diagnostic compile definition separately binds retained source107 manifest SHA-256 `09ae737a09623ff15fc4b1d13f8a96cb14e5d87e0c245cb7b1deb70bcf145d97`. Overlay 109's budget implementation is required for the checker to compile distinct 16/32 material descriptors. The exact point inputs are source-370's compiled package, rejected conditioning candidate nodes, accepted state export, run manifest and conditioning manifest; hashes are embedded in the checker and repeated in the output ledger. The medium source asset is `matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter`, SHA-256 `76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9`.

## Commands

After integrating overlay 109 and this overlay, build only the bounded target:

```sh
cmake --build <build-dir> --target numi-matter-cardboard-source370-material-point-check -j2
```

CPU admission (hashes and two private compiler descriptors; no Metal dispatch):

```sh
<build-dir>/matter/numi-matter-cardboard-source370-material-point-check \
  --cpu-admission \
  /Users/home/cardboard-implementation-20261014/solver/cached-true-residual-certification-coupon-370 \
  /Users/home/cardboard-implementation-20261014/snapshots/source-selected-capture-tail-reuse-108/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter
```

A native one-point dispatch, if separately authorized and scheduled by the root GPU owner, uses:

```sh
<build-dir>/matter/numi-matter-cardboard-source370-material-point-check \
  --native \
  /Users/home/cardboard-implementation-20261014/solver/cached-true-residual-certification-coupon-370 \
  /Users/home/cardboard-implementation-20261014/snapshots/source-selected-capture-tail-reuse-108/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter \
  <new-output-directory>
```

The native mode writes source-bound raw float32 words and status/residual/update details for both budgets. It performs zero physical Matter transactions. It does not itself impose a pass verdict on projection, stress, or tangent outputs.

## Validation recorded here

- Host Objective-C++ syntax-only compilation passed with `-Wall -Wextra -Wpedantic -Werror` and headers from the 109 candidate plus frozen 108/source headers.
- The native target has not been built, the shader has not been compiled in this overlay, and no GPU dispatch was run by this package author. Root should build and run CPU admission before scheduling the native point check.
- This is a diagnostic instrument, not material calibration, a full-coupon replay, or a physical qualification result.
