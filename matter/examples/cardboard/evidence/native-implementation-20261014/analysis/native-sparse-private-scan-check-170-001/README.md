# Native sparse private-contact scan checker, source170

This package builds the source170 standalone differential checker against the exact O3 linked Metal library. The runner is preflight-only by default. It dispatches Metal only when invoked with `python3 runner.py --execute`; no GPU execution has occurred as part of this package preparation.

## Bound source

The checker source is copied byte-for-byte from `hierarchical-sparse-private-contact-scan-overlay-170-001/candidate/matter/tools/cardboard_sparse_private_contact_scan_native_check.mm`. The copied metallib is the source170 `NumiMatter-O3-deps.metallib` artifact. `native-170-command.json` binds the source overlay manifest and patch hashes, checker hash, metallib hash, compiled binary hash, exact compile argv, and launch limits. The runner verifies all source-overlay candidate/base hashes before it permits preflight or explicit execution.

## Checker coverage and memory bounds

`inputs/gpu-case-inventory.json` lists the 49 native differential cases from the checker source: 42 small block-boundary cases, three capacity boundary controls, alternating and deterministic-random multi-block cases, a two-environment healthy-overflow/prefailed-partial case, and the shipping pair-domain case. Together they cover 6,366,845 environment-pair slots. The shipping-domain case has 6,335,568 pairs, a 65,536 candidate cap, and 24,749 blocks.

The compiled ABI helper reports a maximum single-run checker Metal-buffer footprint of 25,703,664 bytes. It does not include candidate contact-sample buffers: the checker does not allocate them. For comparison only, a 65,536-entry runtime sample buffer would be 10,485,760 bytes at the compiled 160-byte sample ABI. The checker also keeps three CPU output-vector copies at its largest case, estimated at 76,813,404 bytes. These are bounded accounting values, not measured unified-memory peak or a performance result.

## Validation performed

- Objective-C++ checker compiled with C++17 and warnings-as-errors against the copied metallib path.
- ABI sizes were compiled from the actual source169 shared header.
- The checker was run only with `--cpu-only`: three CPU contracts passed and it printed `metal_kernels_executed=0`.
- Seven runner guard tests cover default no-dispatch behavior, explicit execution routing, the exact preserved stopped LLDB target, mismatched stopped-process identities, active-checker rejection, reserve-file allocation, and case/ABI inventory.

The native Metal differential has not been executed and no speedup, full-coupon result, shipping admission, or physical qualification is claimed. Preserve the compile/runtime failures and native outputs from any later explicitly approved run.
