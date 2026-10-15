# Angular free-BOX native checker — source066 package

This package contains the standalone native-kernel checker for the angular free-BOX contact path. It is an executable test artifact, not a solver qualification or cardboard-assembly result. The checker performs zero FEM time steps and does not constitute a shipping-box run.

## Source binding

The checker and CMake patch are bound to the immutable `source-armijo-applied-alpha-diagnostic-integrated-066` snapshot at `/Users/home/cardboard-implementation-20261014/snapshots/source-armijo-applied-alpha-diagnostic-integrated-066`, parent revision `5fd4486ea2bcbe903782509a769ce3935b790e55`. `provenance/frozen-source-066.json` is the snapshot's full frozen-source manifest. `manifest.json` records its SHA-256, the exact hashes of the required base files, and the checker/patch hashes. The snapshot directory is external source evidence and is not copied in full here.

To verify package files and the complete source066 file map, run:

```sh
python3 verify_package.py
```

The strict Objective-C++ syntax check was run against headers from source066 with Apple Clang 21.0.0.123.102, arm64, `-Wall -Wextra -Wpedantic -Werror`; the exact command and empty output are retained under `validation/`. The metallib path in that check is only a compile-time string. This package does not contain a newly built executable or native run receipt.

## Integrating and building the native target

Use a disposable source copy or the next source snapshot; do not modify the retained source066 snapshot. Copy `checker/cardboard_free_box_angular_native_check.mm` to `matter/tools/` in that copy, apply `patches/add-angular-native-check-to-source066.patch` from the source root, configure it using the project's ordinary Matter/MetalRobo build, then build the explicit target:

```sh
cmake --build <build-directory> --target numi-matter-cardboard-free-box-angular-native-check
```

The CMake patch links the existing Matter runtime, depends on the Matter shader target, and registers the native check under CTest. Running the executable requires a Metal device and the matching `NUMI_MATTER_METALLIB`. No native execution was performed while preparing this package; the parent task owns GPU scheduling.

## Required native gates

The checker calls production `nm_contact_validate_fem_rigid_armijo_path`, `nm_contact_evaluate`, `nm_fgmres_apply_primal_rigid_contacts`, `nm_project_primal_free_rigid_candidate`, and `nm_fgmres_apply_free_rigid`. Before it can report PASS, a native run must show:

- Direct production sample agreement with an independent FP64 source-equation oracle for gap/witness/normal, normal velocity, inverse response, stiffness, scalar force and impulse, and radial curvature. Each sample prints raw and projected curvature, the native `barrier.y/dt²` curvature, and whether the source equation's sub-floor or nonnegative projection applies.
- Face, edge, and corner cases with positive/negative/mixed signs, fixed-seed basis and random generalized directions, all five epsilon/error pairs, the refinement sequence, a measured same-state repeat floor, and resolved finite-difference agreement at the stated `1e-3` gate. An unresolved sequence is reported as `INCONCLUSIVE` (exit 3); a resolved mismatch is `FAIL` (exit 2).
- The separate production free-rigid inertia action versus projected FP64 residual differences over basis and fixed-seed directions, with the same resolved/inconclusive distinction.
- Production validator cases for rotating end-clear/mid-path-hit, separated miss, offset proxy radius, and distinct physical-time versus Armijo-alpha paths.

A native run is required to bind actual device sample values. The CPU-transcribed values below are sanity expectations only; they are not native measurements or passing evidence.

| Checker case | Gap | CPU oracle force | CPU oracle raw radial curvature |
| --- | ---: | ---: | ---: |
| `face_pos` | 2.000 mm | 11.7125 kN | 5.0334 MN/m |
| `face_neg` | 2.000 mm | 11.7116 kN | 5.0331 MN/m |
| `edge_mixed` | 2.8284 mm | 8.6927 kN | 2.6846 MN/m |
| `edge_mixed_2` | 2.8284 mm | 8.6932 kN | 2.6847 MN/m |
| `corner_mixed` | 3.4641 mm | 7.2434 kN | 1.9553 MN/m |
| `corner_pos` | 3.4641 mm | 7.2428 kN | 1.9552 MN/m |

These estimates use the checker-authored 10 ms step, 12 mm thickness, node/body inverse masses, rotated inverse inertia, and its constructed 0.5 m/s separating normal speed. The corresponding impulses are force times 0.01 s. The native lines report both oracle and actual Metal values, including exact FP32 gap and projected curvature; use those for any measured conclusion.

## Retracted curvature diagnostic

An earlier diagnostic incorrectly suspected that raw radial barrier curvature was negative/clamped. That claim is retracted: the transcription had `4*s²*second` where source066 uses `4*s*second`, with `s=d²`. Evaluating the exact source expression gives positive raw curvature for the checker geometries. With thickness `t=0.012 m`, the intended gap ratios are `1/6`, `sqrt(2)/6`, and `sqrt(3)/6`; their raw `radial/k` values are respectively `0.01014790`, `0.00541262`, and `0.00394252 m²`. Thus the checker samples do not support a claim that the source curvature was clamped. The next native run must compare actual force and radial values against the oracle before any contact/preconditioner diagnosis.

## Current package status

`SOURCE_AND_SYNTAX_SEALED_NATIVE_GPU_RUN_PENDING`. The checker and patch are hash-bound, and syntax passed against source066. The CMake registration patch is ready for integration. Native sample/FD gates remain unexecuted here.
