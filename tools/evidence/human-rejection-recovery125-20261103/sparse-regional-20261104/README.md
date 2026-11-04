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
| Reusable prefix 095 | 4 accepted, bitwise geometry/physiology versus 093 | 17.532 | 72.618 |
| Direct coefficient addressing 097 | 4 accepted, bitwise geometry/physiology versus 095 | 10.023 | 42.182 |

The 095/097 pair measures a 42.83% GPU-duration reduction within the experimental sparse path. This is one exploratory pair, not three matched pairs or a sustained-throughput qualification. It remains slower than production. The 097 accepted-loop rate is 0.00000148167x; production 086 measured 0.00000246089x. No 0.1x result is claimed.

Changing preconditioners changes solver rounding. The sparse path has not established the requested complete force/strain-energy engineering comparison against production. The exact execution changes within the sparse path have bitwise fixture and short native-state comparisons. New settings are not promoted into the production launch.

## Failed preparation and preservation

089 failed the discarded preparatory basis's region budget. Its replacement defers that basis while retaining final source admission. 090 was deliberately stopped before stepping to repair missing indirect-command resource residency. 091 exposed the regional-only prefix admission gap; the supported graph extension has dedicated active/completed/failure fixtures and native 095 coverage. Failed trial receipts and source recovery remain retained.

Each trial includes exact source/configuration pins and losslessly compressed original stdout/stderr with original hashes. One final source archive plus per-trial patches restores all 39 pinned source files; every restored file was independently checked against its trial manifest. Full native geometry remains on SSH Numi, with hashes and verified deduplication recorded in compact comparisons. No captured geometry or old sources were silently replaced.

The current fine source still contains severe sliver conditioning. Four accepted roots do not establish supported equilibrium, breathing or heartbeat duration, sustained anatomy/contact quality, native presentation, clinical validity, or completion. The subsequent 16-root convergence extension is reported separately, including any rejection. Larger timesteps and earlier unsuccessful solver variations are not repeated here.
