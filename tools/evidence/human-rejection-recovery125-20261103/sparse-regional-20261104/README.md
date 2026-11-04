# Exact sparse regional mechanics, 4 November 2026

This increment makes the existing full-coordinate Human regional preconditioner fit its checked storage budget. It does not reduce mechanical coordinates, change anatomy or material laws, relax acceptance, or change the 15.625-microsecond step. All new runtime choices remain opt-in. The faster production configuration remains available.

## Implemented

- Compile coefficient CSR directly, without first allocating the rejected dense extent. Validate source labels, exact anchors, canonical regional supports and node/mode incidence. Preserve coefficient values, cross-region projected mass terms, ancestry and actual-byte capacity limits.
- Whiten regional blocks on their support, using the existing numerical method; validate the resulting full projected mass matrix. Dense reference storage remains available for compatibility and comparison.
- Build tetrahedron support in one source-order pass. The source compiler and runtime consume the same checked sparse representation. GPU lookup metadata and ICB residency are explicit.
- Stream the exact regional FEM off-diagonal action into its existing projection instead of retaining every element/mode action. The original and streamed paths share the material-action evaluator and reduction order. The same-node contribution is removed by the same arithmetic to avoid double-counting.
- Admit reusable regional-only prefix commands with explicit configuration restrictions; retain the original graph-Schwarz path and mandatory coupled true-residual certification.
- Use known regional row ordinals and the validated node/mode coefficient map instead of repeated searches. This changes addressing, not coefficient arithmetic. Version-one lookup compatibility is covered by the fixtures.
- Defer the preparatory skin basis only when the final registered-volume installation replaces it. Source ancestry checks still run; the final physical source, partition, whitening and budget checks remain mandatory.

The tested fine scene has 163 regions, 1,956 modes and 315,197 source rows, including the skin/fat and passive optic sources. Its compact coefficient vector has 4,895,244 float4 entries (78,323,904 bytes), versus a dense capacity of 616,525,332 entries (9,864,405,312 bytes). Streaming omits 3,901,802,368 bytes of intermediate element/mode actions. These are storage reductions, not claimed simulation speedups.

## Tests and measurements

The final native FGMRES contract passes 916,759 assertions. Compiler checks cover dense/sparse bitwise coefficients, projected mass, whitening, B/B-transpose work and source admission. Metal fixtures cover two environments, dense and sparse lookup, partial and empty tiles, invalid material caches, finite-difference tangent products, explicit ICB residency, and active/completed regional-only prefix commands. The streamed tangent matches its stored reference bitwise in eight scenarios. These fixtures do not qualify full-Human sustained motion.

All native measurements ran on SSH Numi: Apple M6 Mac18,5, 16 GiB, macOS 27.0 (26A425). Each short trial requests four roots, totaling 62.5 microseconds of simulation, with the same timestep, full coordinates and final physical checks.

| Trial | Result | Mean GPU seconds/root | Accepted-loop seconds |
| --- | --- | ---: | ---: |
| Existing production 086 | 4 accepted | 5.823 | 25.397 |
| Streamed sparse 093 | 4 accepted | 17.530 | 72.144 |
| Prefix requested 095 (see admission correction below) | 4 accepted, bitwise geometry/physiology versus 093 | 17.532 | 72.618 |
| Direct coefficient addressing 097 | 4 accepted, bitwise geometry/physiology versus 095 | 10.023 | 42.182 |

The 095/097 pair measures a 42.83% GPU-duration reduction within the experimental sparse path. This is one exploratory pair, not three matched pairs or a sustained-throughput qualification. It remains slower than production. The 097 accepted-loop rate is 0.00000148167x; production 086 measured 0.00000246089x. No 0.1x result is claimed.

Changing preconditioners changes solver rounding. The sparse path has not established the requested complete force/strain-energy engineering comparison against production. The exact execution changes within the sparse path have bitwise fixture and short native-state comparisons. New settings are not promoted into the production launch.

## Failed preparation and preservation

089 failed the discarded preparatory basis's region budget. Its replacement defers that basis while retaining final source admission. 090 was deliberately stopped before stepping to repair missing indirect-command resource residency. 091 exposed the regional-only prefix admission gap; the supported graph extension has dedicated active/completed/failure fixtures. A later source inspection found that row equilibration kept native 095 on its direct path; native reusable-prefix execution was not established by that trial. Failed trial receipts and source recovery remain retained.

Each trial includes exact source/configuration pins and losslessly compressed original stdout/stderr with original hashes. One final source archive plus per-trial patches restores all 39 pinned source files; every restored file was independently checked against its trial manifest. Full native geometry remains on SSH Numi, with hashes and verified deduplication recorded in compact comparisons. No captured geometry or old sources were silently replaced.

The current fine source still contains severe sliver conditioning. Four accepted roots do not establish supported equilibrium, breathing or heartbeat duration, sustained anatomy/contact quality, native presentation, clinical validity, or completion. The subsequent 16-root convergence extension is reported separately, including any rejection. Larger timesteps and earlier unsuccessful solver variations are not repeated here.

## Sixteen-root convergence extension 099

The unchanged 097 binary subsequently accepted all 16 requested roots (0.25 milliseconds) with every final status zero. It used 39 active Newton solves and 624 active Krylov columns. The earlier production frontier rejected root 13; this candidate accepts that root and completes the 16-root trial. First-four-row physiology and surface audit fields match 097 exactly. This removes that observed prefix rejection without changing tolerances or increasing iteration budgets. It does not establish later continuation.

The accepted loop took 172.115 seconds, 0.00000145252x; initialization is separate. The preconditioner has useful convergence behavior but still fails the performance objective. Full native geometry/source hashes, command and compact active-work records accompany the trace. The 10-second, 60-second and 330-second supported qualification, complete force/strain-energy comparison and native presentation remain open.

## Follow-up under validation: completed regional builders and row equilibration

Whole-stage profile 101 recorded 16 regional builder calls for two active Newton solves. The sparse block tiles and fused tangent tiles accounted for 5.748 seconds of selected GPU spans. The complete step was 10.017 GPU seconds, versus 10.004 seconds for the uninstrumented first root of 097 (one pair, 0.13% perturbation). These measurements identify work actually executed; no scaled phase sampling is used.

The 102 source follow-up uses the existing healthy full-nonlinear completion latch to zero only intermediate regional scratch-builder dispatches. First and final passes and diagnostic captures stay full. It also extends the regional prefix to the already-supported physical-input unscale used with row equilibration, retaining resource residency and per-command barriers. Native 104 accepted all four roots. All ten emitted CSV files and both captured geometry/state endpoints match 097 bitwise. The log confirms the row-equilibrated preconditioner prefix actually executed (15 prefix commands). The accepted loop decreased from 42.182 to 33.492 seconds (20.6%); mean GPU duration decreased from 10.023 to 7.858 seconds/root (21.6%). This is one exploratory pair and remains slower than production 086. The new rate is 0.00000186614x, not the 0.1x objective. The native FGMRES suite passes 916,759 assertions and the build passes.

Trial 103 was resource-stopped by the unchanged 12-GiB disk guard after 190.811 seconds, before an accepted physical step; it has no valid performance result. Its material census shows zero stateful tetrahedra and zero eligible shared-15-state material work. That proposed optimization is dormant for this Human. The retry follows verified content-preserving output deduplication, whose ledger records 3,855,876,096 allocated bytes recovered without deleting evidence.

Trial 100 was a preparation-only failure: its one-root count conflicted with the retained four-root accepted-Q audit bound. Trial 101 corrects that bound. Their exact logs/configurations remain beside the successful profile. The subsequent 32-root extension is still running and is not covered by the four-root acceptance claim.
