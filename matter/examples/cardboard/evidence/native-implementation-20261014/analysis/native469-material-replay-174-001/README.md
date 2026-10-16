# Native469 material replay checker (CPU-prepared, not dispatched)

This package adds a bounded native diagnostic executable for replaying the exact final-Newton capture from the retained failed source469 run. It binds the captured env0 finite-element rows to the exact compiled package, run manifest/receipt, source-fit parameter table, selected FP64 reference arrays, and separate MP fallback roots. It plans 1,920 center evaluations, 17,280 center tangent-basis actions (nine columns for every tet), and 9,600 evaluations covering center plus four captured endpoint states. Additional selected epsilon, trace, repeatability, and reference checks are kept in separate batches. The batch outputs compare production Metal calls with capture/reference data; they do not alter the solver or accepted history.

## Source and build binding

Base snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-warm-cgls-refinement-169`

- Frozen-source manifest SHA-256: `f5054091c27a7e1add41630aae1b45ee6ecc3e6d4112ee85c931cc4c70f44ea4`
- Existing compile definition `NUMI_CARDBOARD_BASE_SNAPSHOT_SHA256`: `769a5e34c1d919293cb263fd27fbd041a482d22016c2818eec41d44e47935752`
- Source169 helper copy is byte-identical: `865a7399791390c3a1d27f9e8d70bb0aa9a5f8e52ebec14cd350a0c3fe258b81`
- `common.metalinc` and the diagnostic `.metal` source are byte-identical in source169 and source157, as recorded in `source-overlay.json`. The independent MP/FP64 references originate in source157 and are not relabeled as source169 output.
- The CMake patch adds `numi-matter-cardboard-source469-material-replay-check` and depends on the existing source-fit diagnostic shader target. The patch is against source169 `matter/CMakeLists.txt` SHA `e1d9fef5b4e2e61c1caa8557749dc889a0581d2173aa211990ebdecb35adda26`; its dry-run/apply check passed.

## CPU evidence

Strict Objective-C++ syntax (`-Wall -Wextra -Wpedantic -Werror`), direct object compile, and direct link against source169 compiler archive passed. The independent reference exporter passed and emitted 100 selected rows over 20 tets, including 17 MP roots. Checker CPU admission reports 1,920 env0 centers (1,728 paper and 192 starch), all 9,600 captured center/side rows, all 17,280 center basis rows, and the bounded selected/epsilon/repeatability counts. Both checker and exporter reject malformed SHA-256 bindings. Commands, stdout/stderr, and the CMake dry-run are retained under `validation/`.

The origin run receipt is a completed run with exit code 2 and unchanged source inputs. Its rejected status remains part of the input provenance; this package does not turn it into a successful solver run.

## Limits

No native Metal execution occurred here. The executable was compiled and its CPU admission path was run only. Its future `--native-469` mode invokes diagnostic kernels against captured material points, not a physical transaction. Repeat-identical evaluation differences are labeled repeatability only, not an FP32 accuracy/noise floor. Selected MP references are separate high-precision references for specified rows; they do not qualify the full material law, all derivative directions, mesh convergence, a complete solver cycle, or physical cardboard behavior.

Transient helper-include and JSON-delimiter defects caught before sealing are recorded in `validation/revision-notes.md`; the superseded one-line helper variation is retained as a patch.
