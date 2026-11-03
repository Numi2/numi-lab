# Active cavity operator and completed contact incidence, 2026-11-03

Two opt-in execution changes preserve the current equations, precision, timestep, physical state and final acceptance checks:

- `NM_VASCULAR_COOPERATIVE_CAVITY_OPERATOR=1`: one SIMD group per vascular equation shares compensated cavity face evaluation. Lane zero accumulates in the original face order. All other equations retain their original evaluator. Both the direct and cached operator ICB paths select the new kernel. No cavity geometry cache or physical state is added.
- `NM_CONTACT_COMPLETED_INCIDENCE_REUSE=1`: suppress the whole main-candidate incidence build only after the existing completed-geometry admission, a finite nonlinear convergence latch, exact-zero geometry proof, and healthy status. It retains the existing CSR; compaction and fresh friction still run. Trial contact and final full narrowphase/incidence/residual/certification never inherit the gate. One 16-byte transient indirect-dispatch buffer is added. Its pipeline is not created when disabled.

These remain default-off. Contact reuse requires the existing immutable single-environment FEM-only completed contact cache and exact-zero geometry policy. No admission is bypassed.

## Validation

- Existing native FGMRES contract: 916,531 assertions. Its check executable remains an explicit build target.
- Focused completed-incidence dispatch: 189 assertions, including direct/indirect shapes, active/completed/failed/NaN/infinite states and exact proof flags.
- Cooperative vascular JVP: 36 cases, two environments, ordinary and compensated paths, multiple face blocks, signed/cancellation directions, low words, completed/failed/NaN gates, invalid orientation and nonfinite geometry. Valid full output/canary arrays are bitwise equal; invalid geometry is rejected.
- Enabled-path moving-wall cavity regression: 32 accepted steps in two environments; replay and reset bitwise, isolated rollback passed. This is a synthetic moving-wall interface check, not whole-human motion qualification.
- Final integrated four-step capture: endpoint bytes 833,232,924, SHA-256 `66c476e14b867eb9a19039ec112750a3d517a18ce039b24bdb5181fec7ac24c7`, bitwise equal to the published completed-linear-builders-063 reference.
- Every measured run: four accepted steps, status zero, 32 active Krylov columns and two active Newton begins per step; bitwise physiology CSV and accepted common-coordinate/cardiac-coordinate files. The quiet trials do not claim uncaptured full force/velocity parity.

Exact execution source pins, source bundles, configuration, binary/device metadata, command, status traces and test logs are retained here. Execution owner checkout is `/Users/numi/Work/skin-fat-native-integration-209`; its base is f75955f08d6edf0f85027eac7614017ac9fbc996 plus the hashed retained working source. The previous publication baseline is fafb8c04d. Source archive aliases reuse canonical identical tar payloads. Large full endpoint stdout stays losslessly retained on Numi, rather than becoming a Git blob.

The actual execution-source recovery archive is `execution-source-recovery.tar.gz`, 13,391,987 bytes, SHA-256 `092694f84f8acd9bc4dd73714c8539be3a106325201a6280b22230c3849edee3`. Restoration was tested: 1,436 base code files, the tracked patch (55 dirty files), and 64 source overlays reproduce all 282 captured build-source inputs. The archive contains the actual execution checkout's source; it excludes historical evidence, runtime assets, and the Apple SDK, which are pinned separately. See `source-recovery-verified.json` for extraction order and `archive-execution-source-075.py` for construction and restoration checks.

The publication worktree matches 25 of 28 source pins. Its root CMake file, Matter CMake file, and native probe app differ from the retained execution baseline; these differences are preserved in the as-tested recovery archive, not silently merged into main. The five new optimization source/check files match exactly. See `publication-source-parity.json`. Source snapshots and native accepted-step receipts do not promote generic unqualified physical-admission or source-import metadata to whole-human qualification.

The dependency ordering retains the existing serial compute encoders and explicit producer barriers; no synchronization barrier was deleted. See Apple's [resource-synchronization documentation](https://developer.apple.com/documentation/metal/resource-synchronization).


## Matched measurements

Apple M6 Mac18,5, 16 GiB, macOS 27.0 (26A425). One GPU run at a time. No selected-kernel or stage instrumentation in the six quiet runs. Order A/B, B/A, A/B. Same binary and physical configuration; A disables these two changes and B enables them. Existing accepted optimizations remain on in both.

| Run | Mean GPU step ms | Accepted loop wall s | Whole process elapsed s |
| --- | ---: | ---: | ---: |
| final-combined-capture-068-retry1 | 5840.119 | 30.979 | 216.117 |
| quiet-control-069 | 6340.175 | 27.674 | 205.599 |
| quiet-combined-070 | 5834.988 | 25.461 | 200.621 |
| quiet-combined-071 | 5890.813 | 25.555 | 200.575 |
| quiet-control-072 | 6350.465 | 27.678 | 215.662 |
| quiet-control-073 | 6336.565 | 27.573 | 200.572 |
| quiet-combined-074 | 5830.104 | 25.401 | 200.620 |

Median paired GPU-duration reduction: **7.97%**. Median paired accepted-loop throughput gain: **8.55%**. Individual pairs and all raw step durations are in `matched-measurements.json` and per-run compact results. Do not multiply this gain by hypothetical future gains.

Each run advances only 62.5 microseconds (four 15.625 microsecond steps) through the initialization/release prefix. Optimized accepted-loop real-time factors span **2.4457256e-06 to 2.4605726e-06**. Whole-process times additionally include asset preparation and export. These are not supported resting motion, sustained throughput, active-Brain qualification or native presentation measurements. **The 0.1x target remains unmet.** Numerical identity here does not establish physiological or clinical validity.

## Cleanup boundary

Production builds omit check executables from ALL only with BUILD_TESTING=OFF; explicit check builds and testing-enabled ALL remain available. Routine timing defaults disable the 833 MB endpoint text dump, retaining it for explicit fidelity captures. Source archives and byte-identical generated outputs are shared only after hash verification. Full stdout compression records the original SHA, compressed SHA and restore command. The launcher scans stdout with a bounded tail instead of loading the full 833 MB text into RAM. Routine early CPU sampling and GPU-entry VM-map sampling are disabled; disk/swap resource guards remain unchanged. The first final-source capture (068) was resource-stopped during text export at the 12 GiB guard after four accepted physical steps. Its partial endpoint is explicitly unqualified and retained with its resource receipt. The retry completed after verified duplicate sharing and lossless log compression. Anatomical assets, sources, failed trials, provenance and correctness gates are preserved. The active build tree itself is only 138 MB; its 25 MB of check executables were retained. Available disk after cleanup was 20,155,011,072 bytes (18.77 GiB, 20.16 GB decimal); the resource summary retains per-run disk and additional-swap peaks.
