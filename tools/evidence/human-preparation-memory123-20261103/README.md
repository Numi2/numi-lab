# Human preparation memory and native contact work: checkpoint 123

**Partial result. The complete current skin/fat scene has zero accepted physical steps. 0.1x, supported rest, native motion, and presentation throughput are not qualified.**

All execution occurred on SSH Numi: Apple M6, Mac18,5, 16 GiB, macOS27.0(26A425). The execution checkout is `skin-fat-native-integration-209`, HEAD `f75955f08d6edf0f85027eac7614017ac9fbc996` with retained dirty work. Its source pins, binary/library/shader hashes, configurations, exact commands, timing and failed traces accompany each run. This is not a claim that a clean published main passed qualification.

Implemented and tested

- Preparation: avoid a complete authoring-world clone; move NPZ payload allocations; map the compressed archive; release import, support, material-default and authoring scratch at last use. A bounded exact-coordinate/face digest cache reuses only successful repeated exact intersection sweeps; finite, ancestry, topology, orientation, closure and volume checks still run. Failed admissions never cache.
- Metal uploads: private upload staging is bounded to 32 MiB, completed batches release scratch, and initialization remains transactional. Existing active/inactive attachment/contact coupons pass at both 32 MiB and 256-byte staging, including bitwise rejected-state/cursor/history checks. Optional allocation-phase telemetry separates physical/operator/contact/Krylov storage.
- Contact: optional `NUMI_HUMAN_CONTACT_ROW_CACHE=1` combines per-row validation with a current-pose geometry query. Source-order grouped force assembly reuses its force. Audits and wrench reductions skip only finite rows certified inactive by that same encode; active/invalid rows retain the original query and rejection. Scratch is overwritten every encode, with existing ordered encoder dependencies. Filtered/reordered/unsupported layouts retain the original path. Default remains off pending broader accepted-motion qualification.
- Native launcher exposes Matter's existing bounded FGMRES budgets up to 1024 in multiples of16; defaults, restart 16, residual certification and physical acceptance are unchanged. Increasing the budget did not cure this scene's convergence.

Measured evidence and limits

| Measurement | Result | Scope |
|---|---:|---|
| Import scratch release, trial 004 | footprint5.271 ->2.940GB |2.331GB released at last use inside one run; not a matched whole-program memory speedup |
| NPZ reader, three alternating pairs | peak RSS89.19-89.21 ->50.58-50.59MB |about43.3% lower isolated reader RSS, matching payload count/checksum; not whole-scene throughput |
| GPU allocation, full candidate arena |9.93GB |device recommended working set12.71GB; guard retained |
| GPU allocation with contact cache |10.17GB |247,258,832bytes additional derived scratch |
| Matched failed attempt 005 ->006 |51.835 ->31.742seconds |single instrumented pair,1.633x; identical64-column residual trace/rejection; **not accepted-step performance** |
|256-column solve, trial 007 |true residual6.03974695e-5 |from0.709982038; code10 rejection; no accepted physical time |

The production-kernel cache check covers two environments; zero/tile-boundary/large row counts; elastic and exact triangle-bank contact; active/inactive/malformed samples, attachments and bank ranges; source-order forces; fresh cache after node motion. Fused validation flags match the original bitwise. Maximum force error is1.49011612e-8N (relative1.22301988e-7), within the existing1% comparison gate. Original serial versus tiled audit accumulation reaches relative0.000420067377. Source-order force addition is preserved; the first proposed bitwise-force criterion failed because scratch changes rounding, and that failed trial is retained.

The new CMake test target is `numi-human-contact-row-cache-check`; run it with the production `NumiMatter.metallib`. The execution build has `BUILD_TESTING=OFF`, so its attempted CTest invocation found no tests; it is **not** counted as a CTest pass. The built executable was run directly and passed. No paid CI was added.

Trials and blockers

1.001/002/003 stopped on resource guards before physical execution. Restart cleared the recorded53.4 GiB idle decoder allocation; it is separate from measured runtime allocations.
2.004 reached GPU execution and rejected3,631,871broadphase candidate pairs against469,440capacity. Mandatory rollback retained; no physiology/controller advancement is claimed.
3.005 used a4,000,000skin/fat candidate allowance, retained the device guard, and reached real linear-solve exhaustion.
4.006 isolated the first contact cache against005.007 fused validation/query and used256FGMRES/4Newton budgets; the first solve still exhausted its budget. Later encoded tails are not active Newton solves.
5.008 rejected the default regional preconditioner before GPU execution because36-face chunks exceeded170regions.
6.009 used supported256-face chunks and one source-tet 0refinement patch. Region admission passed, then the compiler rejected dense coefficient rows exceeding its checked 384 MiB budget. The current sparse export occurs **after** dense construction. Do not remove this guard or allocate the dense basis on this 16 GiB host. The required implementation is sparse regional construction through combination, mass whitening and GPU export, with dense-oracle Bq/B^Tf/projected-mass/tangent parity before another full run.

These are short initialization/release diagnostics at 15.625 us, not a settled supported checkpoint. No realtime factor is assigned to rejected steps. The0.1x target remains unmet; longer baseline/intervention/video qualification must wait for accepted mechanics and an accuracy-qualified timestep. Physical geometry, material laws, mass accounting, force ownership, source ancestry and acceptance tolerances were not relaxed.

Storage and recovery

The combined verified cleanup and user restart recovered 24.699686912 GB at the post-restart checkpoint. Subsequent builds/trials and OS paging change current free space; `cleanup-final.json.gz` retains fresh byte counts rather than adding logical archive totals. Current sources, accepted anatomy evidence, role inputs and both required all-child mass proofs remain on Numi. 19 historical batches were independently downloaded, archive/member-hash checked and restore tested before retiring exact old paths. Byte-identical transparent compression also retained paths and hashes.

Private recovery release: https://github.com/Numi2/macmini-evidence-archive/releases/tag/numi-20261103-historical-physics-evidence-memory123

`memory123-native009-source.tar.gz` plus its existing per-file manifest and verification receipt preserves 556 native source/configuration dependencies, with no source retirement. Its SHA256 is `e9d6d3adef4309125d0bc3fd00aa7c001527dd4b63dcd4bf01d3d564efdcbafd`. Apple toolchain source is excluded. Background media analysis was restored after measurements; no task-owned native GPU benchmark remains running.

All included `.gz` files are compressed copies of existing configuration, receipt, trace, patch or log formats. Failed and inconclusive trials remain alongside passing controls. This checkpoint is numerical/software evidence, not anatomical motion, physiological, or clinical validation.
