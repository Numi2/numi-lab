# Source-392 tet-858 native material-point replay overlay

This isolated overlay generalizes the source-370 native point host to one exact source-392 material point. It reads the original compiled package and its rejected-conditioning candidate directly. It does not recompile the medium material, alter the material iteration budget, update a physical step, or modify accepted history. The Metal kernel is the existing `cardboard_replay_source15_candidate_point` from the source-118 paper diagnostic shader; it calls production `tetrahedronDeformation`, `femMaterialDeformationWords`, `tetrahedronVelocityGradient`, `femMaterialInput`, and `evaluateAlgorithmicStressTangent`.

The host is fixed to tet 858 and fails closed on any input digest, dimensions, fingerprint, material, connectivity, frame-policy, timestep, state, or source-build mismatch. It uses the compiled package's material descriptor, parameter words, bytecode and object; the supplied source asset is hash-checked against the run manifest. The accepted material state is read from the retained rollback export and required to be 15 zero words. Both environment node records are read, with the actual point kernel evaluating environment 0.

The retained 392 candidate is `rejected-conditioning-000001/candidate_nodes.bin`, marked `accepted_physical_state=false`, `control_step=0`. The sidecar has no field identifying the exact first-failure Newton generation, so a native point result is an exact replay of this later rejected-conditioning snapshot, not proof that this was the generation that first emitted status 5. No full FEM transaction or coupon qualification is performed.

## Build integration

Apply `CMake-fragment.txt` next to the existing source-370 point target in the source-118 `matter/CMakeLists.txt`. The source-118 build receipt is SHA-256 `da94e6051e767ea4bfba58705ccaf321e5dd97e5e9b40b32715f056676980c9f`; the target depends on the existing `numi_cardboard_paper_shaders` so the point kernel is present in the linked paper metallib.

## Commands

The CPU-only preflight already completed successfully with:

```sh
source392-tet858-point-check --validate \
  /Users/home/cardboard-implementation-20261014/solver/dimensionless-certified-coupon-392 \
  858 \
  /Users/home/cardboard-implementation-20261014/snapshots/source-scaled-material-merit-118/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter
```

The native diagnostic invocation for the parent-owned runner is:

```sh
source392-tet858-point-check --native \
  /Users/home/cardboard-implementation-20261014/solver/dimensionless-certified-coupon-392 \
  858 \
  /Users/home/cardboard-implementation-20261014/snapshots/source-scaled-material-merit-118/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter \
  /Users/home/cardboard-implementation-20261014/native/source392-tet858-point-121
```

The output directory must be new. The host writes all raw 160 float32 result words, F high/low, Fdot, projected state high/low, stress, tangent action and local projection diagnostics. It does not fail merely because the point projection reports invalid; that result is the diagnostic target.
