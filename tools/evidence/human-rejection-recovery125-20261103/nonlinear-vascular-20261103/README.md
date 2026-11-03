# Human work cleanup and rejected solver directions

This increment keeps two exact execution changes: reuse the frozen-FEM cavity constraint's single nonzero vascular derivative while constructing its principal factor, and opt-in compact skin/fat mass capture. No material, force, geometry, mass, timestep, final residual/contact/cavity certificate, or rollback rule changes.

The attempted hydraulic block relaxation between body Newton updates is **removed from production source**. Its exact patch is retained as `vascular-newton-relaxation.unqualified.patch`, alongside failed native trials. The 083 extension rejected after 10 accepted roots (156.25 microseconds) on mechanical nonlinear certification; the previous unmodified solver reached13. The 62.5-microsecond085 trial again rejected its second root even with vascular residual near 5.2e-14. These are failed experiments, not progress in whole-human convergence.

## Keep

- Principal-factor construction evaluates the unchanged cavity derivative arithmetic once per cavity, replacing repeated whole-surface traversal for every vascular column. With FEM fixed, all other vascular entries in that cavity row are exactly zero. Cavity geometry validity is still evaluated; the full coupled JVP, factorization, precision and all final certificates remain unchanged. One 256-byte threadgroup array is added; no persistent state, buffers, settings or public API.
- `NUMI_HUMAN_SKIN_FAT_COMPACT_MASS_CAPTURE=1` omits five routine CSV exports totaling 324,451,350bytes in this scene. All parent mass/closure, source ancestry, positive reference-volume and region/face-label checks still execute. Full exports remain the default. Explicit full-weight/layer/deep-attachment export requests cannot be silently suppressed. The as-tested application block matches publication source; other pre-existing app differences remain in the recovery archive.
- Legacy production algorithms and all unrelated dirty source remain intact. No visual verification was performed.

## Validation and measured limit

Final production source passed the existing 916,531 native FGMRES assertions. The principal-factor comparison uses the actual prior metallib: 7 cases cover ordinary and compensated arithmetic, repeated surfaces, reference-displacement low words, invalid orientation, isolated nonfinite geometry, and capacity rejection. Every valid factor and pivot is bitwise identical; invalid environments remain rejected. The independent 46-row FP64 oracle and 36 cooperative JVP cases pass. Existing 32-step synthetic moving-wall tests pass conservation, replay, reset and isolated rollback. Those are interface regressions, not whole-human motion.

`final-production-native-086` advances four 15.625-microsecond roots with all physical certificates active. Physiology CSV and initial/final 321,323,008-byte geometry packs are bitwise identical to `quiet-combined-074`. Each root uses 32 active Krylov columns and two active Newton begins. Skin/fat parent mass remains 13.939678963460397kg with zero reported nodal mass-closure error; reference source 311,768 nodes and 1,453,640 tets is unchanged.

Apple M6 Mac18,5, 16 GiB, macOS 27.0 (26A425). Final mean GPU time is5,823.383ms/root. The accepted body loop advances 62.5 microseconds in 25.3973 seconds: **0.00000246089x**. Process time 195.571seconds additionally includes asset preparation, compilation and initial exports. This single regression establishes no new whole-run speedup. The earlier six matched quiet runs measured an 8.55% median accepted-loop improvement for cooperative cavity JVP plus completed contact incidence reuse; that gain is retained, not re-counted.

**0.1x, supported equilibrium, sustained full-scene motion, breathing/heartbeat duration, native presentation and five-minute qualification remain unmet.** These release-prefix runs start `balanced=false`, without active articulated support; they must not be represented as supported rest.

## What was already tried

`prior-numerical-trials-review-087.json` links retained commands/source manifests and failure receipts. Sources differ across some trials; counts are historical observations, not controlled causal comparisons.

| Method | Retained outcome |
| --- | --- |
|125 us fine-scene step, `dt125 us` |0/1accepted |
|62.5 us, `034` |1/1accepted; no accuracy or continuation proof |
|62.5 us extensions, `079`, `085` |Both1/4accepted; vascular and then mechanical rejection |
|64 linear budget, `078` |9/16accepted versus13/16at the smaller current budget |
|More Newton work, `020` |5/16accepted,24 active Newton begins in rejected root |
|Thin-cell Schwarz, `029` |10/16accepted |
|PSD contact graph, `036` |8/16accepted |
|Pressure-load continuation, `033` |12/16accepted |
|Initial vascular predictor, `030` |12/16accepted; failed mechanically |
|Interleaved vascular relaxation, `083` |10/16accepted; failed mechanically despite conserved vascular state |
|Assembled xyz operator, `038` |4/4prefix passed; no sustained qualification |
|EW forcing variants, `058`, `062` |Short prefixes passed; numerical parity/acceptance limits retained, no sustained gain promoted |

The separately retained23 October audit also records that250 us trajectories failed the existing force/strain comparison against125 us in the earlier coarse scene, despite the selector-6 repair accepting125 us prefixes. That is historical evidence from a different model; it does not qualify this fine scene.

**User instruction: larger timesteps are retired as the optimization direction. Do not repeat those trials.** Increasing budgets or re-toggling failed local preconditioners also requires a specific new mechanism and a new falsifiable witness.

Current source already contains distributed reductions, active-work/ICB suppression, sorted contact work, shared 15-state material factorization and stateless/native tangent machinery. Their existence is not permission to bypass their individual eligibility/admission rules or evidence that every path is active/useful in Human.

## Structural blocker to address next

The current “sparse” Human partitioned path first constructs a dense mode-major basis. The full skin/fat/optic model exceeds the 384 MiB coefficient admission budget; converting it to sparse afterward cannot fix peak construction memory. The current hybrid coordinate path also excludes the mandatory full coupled true-residual certificate. The retained proposal explicitly records NOT_RUN for that reason. Do not disable the certificate or remove the dense vascular owner to make an experiment run.

Continue inside the existing Matter owner with streamed sparse support/local factors and a coupled preconditioner or algebraic interior/interface condensation that retains the full physical residual. A genuinely reduced coordinate model additionally requires conservative mass/force/tangent mapping and correctly projected certification before admission. None of these structural gains is claimed by this increment.

A refreshed conditioning audit uses the actual current cooked source-reference node/tet CSVs, not the older 1,495,294-tet NPZ. It finds zero invalid source cells under its determinant check, but minimum altitude 1.58434 nm, 11,743 cells below 100 nm and 101,192 edge/altitude ratios above 1000. These are conditioning witnesses, not a new quality policy, a waiver, or proof of the sole cause of nonlinear failure. Source hashes and executable audit code are retained.

## Exact reproduction and resource boundaries

Execution checkout: `/Users/numi/Work/skin-fat-native-integration-209`, base `f75955f08d6edf0f85027eac7614017ac9fbc996` plus preserved dirty source. Publication worktree: `/Users/numi/Work/current-fractional-020-publication-20261101`. The three modified owner files are the application, vascular shader and focused vascular test.

Use the sibling `active-cavity-contact-20261103/execution-source-recovery.tar.gz` recovery recipe first, then extract the **single** final 28-file source overlay `execution-source-final-086.tar.gz`. For a historical trial, apply its small `source-from-final.patch`. Restoration tests reproduce every overlaid file byte for byte for seven historical trials. Do not retain a repeated 28-file archive per trial. Exact commands, 28 source pins, binary hashes, launch settings, device metadata, compact CSVs, tests and geometry receipts are in the per-trial directories. Native assets/large geometry packs remain preserved and pinned on Numi; they are not uploaded or relicensed.

Publication matches 25 of 28 execution pins. The pre-existing rootCMake, MatterCMake and native app differences remain explicitly recorded; the new application block is identical. This is not a claim that main alone reproduces all pre-existing dirty development source.

Only one native GPU job runs at a time. No source/binary mutation occurs during native execution. Trials retain first failures and unchanged 12 GiB disk / 3 GiB additional-swap guards. Duplicate immutable outputs are shared only after exact hashes and file-stability checks; source assets, old evidence and every retained path remain available.
