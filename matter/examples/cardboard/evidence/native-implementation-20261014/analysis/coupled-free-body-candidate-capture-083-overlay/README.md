# Bounded coupled Newton candidate capture overlay

This is an external, opt-in diagnostic overlay for `matter/src/runtime.mm` only. It is based on the exact runtime file from `source-runtime-diagnostic-return-repair-079` (SHA-256 `951cfaeafb3bee22b547ff4d8f9fce2c5c9c226364deecf8f4205e13c742cb56`). The patch is emitted as `runtime.patch`; it is not applied to a shared checkout.

The selector is `NM_FEM_COUPLED_CAPTURE_SELECTOR=<control-step>:<zero-based-Newton-iteration>` and the destination stem is `NM_FEM_COUPLED_CAPTURE_PATH=<path>`. Both are required. The callback writes `<path>.coupled.bin` and `<path>.coupled.json` using exclusive creation. The capture is placed after FGMRES correction measurement and Human direction mapping, before the Armijo trial. It preserves the full assembled FEM right-hand side into the existing private diagnostic buffer before FGMRES reuses `femResidual`.

The payload is limited to 32 MiB and 192 arrays. The larger descriptor ceiling accommodates all 158 unique named array slots, with optional arrays omitted when absent; the byte cap remains the controlling memory bound. It includes accepted/current candidate FEM geometry and displacement-mode words, material histories and programs/parameters, contact/cohesive state and workspaces, rigid proxies/states and borrowed scene/q/v/wrench inputs, the completed coupled correction and candidate state, Human coupling data, solver/operator workspaces, and regional/RAP/graph-Schwarz arrays. Each array gets a byte range and SHA-256 in the JSON. The metadata explicitly says that this is a raw transaction candidate, not an accepted checkpoint.

The scope fails closed for unsupported sizes/features and for nonzero articulated generalized unknowns. It targets a one-environment native free-body coupled solve; it is not an articulated-body capture format. Capture-on/off physical history identity and a successful native selected-state replay remain unverified. The original regional graph matrices are omitted because they can be reconstructed from the captured same-candidate operator; the manifest records that omission.

## Offline inspection and on/off history comparison

`coupled_capture_reader.py inspect <capture.coupled.json>` validates the capture and every array hash, checks byte extents/nonoverlap, and prints an array inventory. It decodes the 80-byte FEM node record and promotes FP32 data directly from the stored little-endian words. `high_low_f64(high, low)` promotes both FP32 words first and adds them in FP64; it does not parse rounded decimal JSON values.

For the root 1201 / Newton 12 run, keep the frozen source/build and all physical arguments fixed. Add only the capture selector and path to the diagnostic-on invocation:

```sh
CAPTURE_TOOLS=/Users/home/cardboard-implementation-20261014/analysis/coupled-free-body-candidate-capture-083-overlay
mkdir -p "$ON_DIR"
export NM_FEM_COUPLED_CAPTURE_SELECTOR=1201:12
export NM_FEM_COUPLED_CAPTURE_PATH="$ON_DIR/candidate-1201-12"
# Run the already sealed native command with its unchanged physical arguments.
python3 "$CAPTURE_TOOLS/coupled_capture_reader.py" inspect "$ON_DIR/candidate-1201-12.coupled.json" \
  --report "$ON_DIR/candidate-1201-12.reader-report.json"
```

Run the same sealed native command in a separate diagnostic-off directory after unsetting both capture variables. Compare the raw production history/checkpoint files selected for that fixture; the comparator excludes `.coupled.bin` and `.coupled.json` files:

```sh
unset NM_FEM_COUPLED_CAPTURE_SELECTOR NM_FEM_COUPLED_CAPTURE_PATH
python3 "$CAPTURE_TOOLS/coupled_capture_reader.py" compare-histories \
  --on-dir "$ON_DIR" --off-dir "$OFF_DIR" \
  --include '*.nmcb' --include '**/*.nmcb' \
  --include '*.csv' --include '**/*.csv' \
  --report "$PAIR_DIR/diagnostic-on-off-history-comparison.json"
```

The comparator checks file presence and byte hashes only. It does not verify that run receipts have identical source, executable, or physical arguments, nor does byte identity turn a failed or incomplete attempt into a passing simulation. Retain and compare those receipts separately.

## Verification

The overlay passed Objective-C++ `-fsyntax-only` using the runtime compile command from `build-cardboard-provenance-083/compile_commands.json` with source083 include headers and `matter/src` added to the include path. Runtime JSON declares 192 descriptor slots for 158 unique named array slots; the byte cap remains 32 MiB. No GPU kernel or physical experiment was run. `verify_overlay.py` checks exact base/patched hashes, patch reapplication, unique array names, required captures and scope guards. `test_coupled_capture_reader.py` checks hash/extent rejection, FP32-to-FP64 promotion, node ABI decoding, and history comparison. The runtime manifest records the source083 snapshot manifest hash used for the syntax check.

To apply manually to a copied source tree whose runtime matches the exact base hash:

```sh
patch -p1 < runtime.patch
```

Do not transplant the entire file onto a later runtime snapshot. Cherry-pick the capture hunks into the intended frozen source and review them against that source before building.
