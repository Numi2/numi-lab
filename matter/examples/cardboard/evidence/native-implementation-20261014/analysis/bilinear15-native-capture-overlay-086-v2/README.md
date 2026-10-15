# Captured source15 native projection point check (overlay 085)

This package adds a bounded native material-point replay for the source-fit bilinear 15-state medium. It uses the production `evaluateAlgorithmicStressTangent` entrypoint from the exact frozen084 `common.metalinc`; it adds no constitutive-law implementation. No solver timestep or physical qualification is inferred from this test.

The selected inputs are eight environment-0 medium tetrahedra from each retained FEM capture: Newton 0 / N0 and Newton 63 / N63. `prepare_capture_inputs.py` verifies both full-capture SHA-256 values, the source material hash, and the reference bindings before emitting only those 16 rows. N0 has a separate mpmath 50/80-digit transition reference; N63 uses the independently replayed FP64 projection/tangent array. Each test row carries the exact high/low deformation words, accepted state, deformation direction, deformation-rate direction, timestep, temperature, captured default outputs, reference outputs, and a branch indicator. The branch indicator reports the reference accumulated-multiplier change; it does not promote a near-transition point into a categorical yield proof.

The checker compiles two material descriptors from the same captured asset. Both use the capture's `paperCompensatedArithmetic` and `genericCompensatedImplicitProjection` flags. The default/off control leaves `genericBilinear15AccurateProjection` disabled and explicitly uses the run's `--material-iterations 32` setting. The opt-in descriptor enables the 15-state policy and must compile with local iteration budget 16. The tool checks source flags, parameter float bits against the captured parameter vector, and those compiled iteration budgets before any Metal command runs. Its `--cpu-admission` mode exercises those source and policy bindings without creating a Metal device or submitting a GPU command.

For each policy and each point, the checker evaluates the same row twice. It emits raw state/stress/tangent, projection status and iterations, comparison metrics, and the maximum component-wise difference between repeated evaluations. That repeated-run difference is only a nondeterminism observation; it is not a representability or accuracy bound and never substitutes for an error gate. Default/off outputs are compared with the captured production point, and opt-in outputs with the independent FP64 reference. Nonzero reference norms use the 1e-3 relative gate. An exactly zero reference norm is marked inconclusive because this checker has no independently established absolute tolerance. The checker emits one case record for every input, even after per-case projection, nonfinite, comparison, or instrument errors, then emits final counts and exits nonzero if any case failed or remained inconclusive. Nonfinite GPU outputs are serialized as JSON `null` alongside their exact raw float32 bit words. This test does not change the authored global solver tolerance.

## Build and input preparation

Run these commands from this package on the same Apple Silicon host. Building compiles Metal and Objective-C++ only; it does not launch the checker or run a GPU material point.

```sh
cmake -S source -B build -DBUILD_TESTING=OFF
cmake --build build --target numi-matter-cardboard-bilinear15-captured-point-check -j2
python3 prepare_capture_inputs.py --output evidence/captured-point-inputs.json
build/matter/numi-matter-cardboard-bilinear15-captured-point-check \
  --cpu-admission evidence/captured-point-inputs.json \
  source/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter
```

The sealed CPU-admission record is in `evidence/cpu-admission.log`. A negative mutation check changes one captured parameter word and is expected to fail; its retained output is `evidence/cpu-admission-mutation.log`. This proves that the checker rejects a mismatched parameter binding, not that the native point replay passes.

The input exporter is intentionally bound to the original capture and reference paths under `/Users/home/cardboard-implementation-20261014`. It refuses changed hashes and refuses to overwrite its output.

## Native command for root-owned execution

The checker is not registered as a CTest because it needs the retained captures and a physical Apple GPU. When the serialized native slot is available, run:

```sh
build/matter/numi-matter-cardboard-bilinear15-captured-point-check \
  evidence/captured-point-inputs.json \
  source/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter \
  > evidence/native-captured-point-result.jsonl
```

The v2 output binds the exact084 `frozen-source.json` SHA, the input JSON SHA, and the medium asset SHA. It attempts all four evaluations per case (default twice and opt-in twice) and emits 16 case records plus a header and end record. This package intentionally contains no native output because no GPU execution was performed here. Root-owned native execution must use this exact checker, input JSON, source asset, and frozen084-based overlay.

## Limits

This tests local projection, first Piola stress, and one captured algorithmic tangent direction at selected captured states. It does not test all 15-state directions, all elements, native FEM residual assembly, global convergence, contact, damage overlays, full-cycle history, calibration, physical specimens, or box assembly. Passing would support only these pointwise replays. A failure or unresolved near-zero comparison must remain visible and must not be averaged away.
