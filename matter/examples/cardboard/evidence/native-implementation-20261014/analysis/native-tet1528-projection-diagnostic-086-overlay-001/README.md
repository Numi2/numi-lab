# Bilinear 15-state captured-point diagnostic (frozen source 086)

This external overlay adds a bounded native Metal diagnostic for the retained root-0/Newton-0 FEM capture. It evaluates direct material stress, the production algorithmic tangent for the captured direction, and a direct `projectMaterialState` trace at env0/tet1528 in the liner. The batch covers all 3,456 liner and medium tetrahedron rows in both environments, under the source-default and accurate-projection policies.

The checker reads an explicit `--expected-capture-sha256` binding and hashes the supplied capture before parsing it. The CPU preflight in this package is intentionally pinned to the earlier source-283 first-control capture (`4decb7be…66b870`). That capture is not asserted to be the exact failed source-288 candidate: source 288 failed at env0/tet1528 with status 5, `J=0.99751335382461548`, zero accepted steps, and compiled local iteration budget 16 (the command's requested budget was 32). The exact-candidate capture from the later failure run is a distinct input and can be passed to the checker with its own independently verified SHA-256. The checker retains the same source/shape/material/parameter checks, so a different or incomplete capture fails closed.

## Scope and trace fields

The checker compiles the exact liner and medium `.nmatter` assets with `paperCompensatedArithmetic` and `genericCompensatedImplicitProjection`. Its second policy additionally enables `genericBilinear15AccurateProjection`; source compilation verifies that the actual local budgets are 32 and 16 respectively. It does not alter authored parameters or the global solver tolerance.

Every point record includes all 65 output float32 words as raw hexadecimal bits, alongside JSON values (nonfinite JSON values serialize as `null`). Selected-point trace output carries raw bits for all 73 summary words and all 5,992 trace words. The diagnostic-only `common.metalinc.patch` stores the scaled local-correction norm in previously unused per-iteration trace slot 5; it does not affect stopping, residual, or line-search logic. The trace header reports whether this metric exists.

The production projection trace has a 16-iteration capacity. Thus the default 32-iteration policy is explicitly marked `unsupported_by_existing_16_iteration_trace_capacity`; the separate direct projector result remains available as a batch diagnostic. Do not treat absence of its iteration trace as a failure or as a pass.

This is an instrument of local constitutive behavior only. It performs no FEM or physical simulation steps and establishes no calibration, full-cycle convergence, box assembly, or physical qualification.

## Build in a writable source copy

Use a writable copy whose `frozen-source.json` SHA-256 is exactly `769a5e34c1d919293cb263fd27fbd041a482d22016c2818eec41d44e47935752`. The frozen snapshot itself is not modified. Add the checker and shader to `matter/tools/`, then apply the two source patches:

```sh
git apply /path/to/common.metalinc.patch
git apply /path/to/CMakeLists.patch
cmake -S . -B build -DBUILD_TESTING=OFF
cmake --build build --target numi-matter-cardboard-bilinear15-capture-diagnostic-check -j2
```

The target builds a dedicated metallib containing the diagnostic kernels. The saved `.air`/`.metallib` and object file in this overlay are compile artifacts, not native run results.

## Commands

The package's CPU preflight verifies the pinned source-283 capture and its 3,456-row paper-material inventory:

```sh
python3 capture_preflight.py
```

The built checker's CPU admission mode compiles the source materials and verifies all capture parameter words without creating a Metal device:

```sh
build/matter/numi-matter-cardboard-bilinear15-capture-diagnostic-check \
  --cpu-admission --expected-capture-sha256 \
  4decb7be2fbfa93ad97c0f2f22cfaca974931269068bed90e98815b09066b870 \
  /path/to/operator-capture \
  source/matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter \
  source/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter
```

Run the selected point first. This command evaluates both policies at env0/tet1528 and emits raw trace output without waiting for the 3,456-point batch:

```sh
build/matter/numi-matter-cardboard-bilinear15-capture-diagnostic-check \
  --native --selected-only --expected-capture-sha256 "$CAPTURE_SHA256" \
  /path/to/operator-capture \
  source/matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter \
  source/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter \
  /path/to/new-output-directory
```

The complete 3,456-row replay is a separate explicit opt-in. It runs after the selected traces and is bounded by the capture shape checks:

```sh
build/matter/numi-matter-cardboard-bilinear15-capture-diagnostic-check \
  --native --all-points --expected-capture-sha256 "$CAPTURE_SHA256" \
  /path/to/operator-capture \
  source/matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter \
  source/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter \
  /path/to/new-output-directory
```

`CAPTURE_SHA256` must be computed from the exact immutable capture being used. The source-283 preflight JSON is not a substitute for verifying a later run's capture binding.

## Validation record and limits

The recorded validation is strict Objective-C++ syntax and object compilation, Metal AIR compilation and metallib linking, Python capture preflight, and `git apply --check` for both source patches against frozen086. All passed. No checker binary was linked from CMake and no native Metal command was submitted in this overlay; the parent task owns native execution. Results remain pending until that exact source-bound run is performed.
