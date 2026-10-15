# Captured EE conservative-advancement Metal diagnostic

This package prepares a standalone native Metal diagnostic for the exact edge-edge rays captured in failed private trials 333 and 334. It compares the frozen production `edgeEdgeCCD` equations with a 16-iteration budget and a 17-iteration budget on the same four packed FP32 cases. It does not change the solver, contact thresholds, or captures, and this package has not dispatched GPU work. This version fixes a manifest-reader bug found before GPU dispatch: the corpus manifest contains nested per-capture `payload_sha256` fields as well as a root packed-payload hash, and the runner now reads the root field specifically.

## Source and input binding

The frozen source root is `/Users/home/cardboard-implementation-20261014/snapshots/source-center-action-twoenv-restart-099`. Its `frozen-source.json` SHA-256 is `5fb22c61c280f576f15677c8cf6dbbc174beceaa225973242f6d53f9c7957446`, with parent revision `35bfb518c542a306f563cc364ef8496837068b9a`. Before generating Metal, the tool verifies all 775 files named in that frozen-source manifest. The contact implementation SHA-256 is `3cef6ae3dbd384d7b26ff6b0eb4ecda6da7a32be74cdea0badc5975eef692132`.

`prepare_capture_cases.py` verifies the two retained capture manifests and payloads before packing anything. It selects root 1201, Newton 12, trials 20 and 21, active candidate slot 331, primitive identities 88/127, and the captured edge-edge endpoints on nodes `[13,14]` and `[28,32]`. It reconstructs the start positions using the captured FP32 node velocity, object timestep, and the same `finish - dt*velocity` relation used by the CPU replay. The generated four-case input has 333/16, 333/17, 334/16, 334/17, and its exact little-endian byte hash is `80eaab55f3d471934382eaec6165c2c5a717abafb94cf08dd75f7bb69c42d3f9`.

The checked-in `captured-input` files preserve those packed case words and their manifest. Their source captures remain the authoritative input; the packer rejects changed capture hashes, selector identity, candidate identity, or triangle-to-node mapping.

## Diagnostic implementation

`generate_metal.py` extracts the exact frozen `edgeEdgeCCD` function definition and emits two separately named clones. The 16-step clone retains the literal 16-iteration loop; the 17-step clone changes only that literal to 17. Both initialize output-only witness fields and record the existing iteration time, closest points, segment parameters, distance, effective thickness, and speed bound. The production function remains present in the included frozen Matter translation unit and is not edited. The generator also verifies the frozen manifest and all 775 listed source files before compilation.

The Metal runner calls both the unchanged `nm_contact_ccd_regression` kernel from the frozen translation unit and the source-extracted diagnostic kernel. For each 16-step case it reports whether the diagnostic scalar result is bitwise identical to the unmodified production kernel. It reports `certified_hit`, `certified_miss`, or `uncertified_fail_closed`, plus per-budget TOI, separation, normal/parameters if hit, final closest-point witnesses, number of iterations, and the bounded iteration trace. The result explicitly marks solver and physics acceptance as unevaluated: CCD classification alone does not accept an Armijo trial or transaction. If compiled instrumentation changes the budget-16 result, the report makes that mismatch explicit; it does not hide it or claim production parity.

## Build and later native run

The build script uses the same Metal flags as source 099: `-std=metal4.0 -O3 -fno-fast-math`. Building the shader and host executable does not execute a kernel.

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/prepare_capture_cases.py \
  --capture-333 /Users/home/cardboard-implementation-20261014/native/center-action-feasibility-bracket-333/private-trial.coupled.json \
  --capture-334 /Users/home/cardboard-implementation-20261014/native/center-action-feasibility-bracket-334/private-trial.coupled.json \
  --output-dir /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/captured-input

bash /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/build_diagnostic.sh \
  /Users/home/cardboard-implementation-20261014/snapshots/source-center-action-twoenv-restart-099 \
  /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/build \
  /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/captured-input/captured-ee-cases.float4.bin
```

After those checks, refresh the package hash inventory with `python3 /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/seal_package.py`.

The host-side admission regression runs without creating a Metal device:

```sh
/Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/build/run_native \
  --validate-inputs-only \
  --metallib /unused/not-loaded.metallib \
  --cases /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/captured-input/captured-ee-cases.float4.bin \
  --cases-manifest /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/captured-input/captured-ee-cases.json \
  --build-manifest /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006/build/build-manifest.json
```

After a GPU worker is explicitly available, the runner command is printed by the build script. It needs the generated metallib, the packed cases, both adjacent source-binding manifests, and can save JSON with `--output PATH`. The sealed manifest records the exact command with `native-result.json` as its output. No GPU result is included in this package yet.

## Checks and limits

After generation/build, `python3 -m unittest discover -s /Users/home/cardboard-implementation-20261014/analysis/private-ccd-metal-compare-006 -p 'test_*.py' -v` passes four CPU checks for source extraction, bounded instrumentation, and exact capture packing. The standalone shader and Objective-C++ runner compiled successfully without dispatching a GPU kernel. Metal compilation reported two pre-existing warnings in frozen source 099's FGMRES kernels.

This is a native diagnostic harness, not a claim that the 17-step candidate is physically correct or that the enclosing Armijo trial becomes feasible. The CPU source-order replay had already found that its FP32 interpolation form can exhaust 16 iterations and resolve the captured near miss on the next conservative-advancement iteration; that replay is not Metal attribution. The native runner must be executed on the captured cases before claiming what the production Metal implementation returns.
