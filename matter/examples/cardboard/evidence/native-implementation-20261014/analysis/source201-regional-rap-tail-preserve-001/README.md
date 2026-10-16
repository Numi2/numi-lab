# Source 201 regional RAP tail-preservation candidate

This candidate derives from immutable source198. It adds restart-only kernels that keep a converged environment's same-Newton augmented RAP matrix, LU/pivots, validity and fallback state unchanged across completed restart tails. The original base RAP kernels and their initial matrix build/factor dispatch remain byte-identical to source198. The candidate also fixes the restart RAP builder launch extent to `environment_count * mode_count * mode_count`; the old source198 host launch supplied only `mode_count * mode_count`, so an E2 launch did not cover environment 1.

The preservation marker is allocated only when restart augmentation is enabled, reset alongside factor validity at the start of each Newton solve, and consumed only by restart-specific wrapper kernels called inside the augmentation branch. The final true-residual and certificate path is untouched.

Validation: source-contract script passed all static checks; the source-bound Release/O3 and Debug/no-explicit-optimization native-checker targets compiled successfully against 817 selected source files. The integrated Metal target is hardcoded O3. A separate full `matter_kernels.metal` compile with Metal `-O0` also succeeded (two pre-existing uninitialized-threadgroup warnings recorded in command output). The checker was not run; there is no native kernel execution or solver timing/parity claim.

Preserved failed attempts: first Release build stopped because noninteractive PATH omitted `/opt/homebrew/bin`; second reached checker compile and failed due global `pIndex` name collision. The successful retry renamed the checker helper. See `source-overlay.json` for full paths, hashes, receipts, and limits.

The patch also updates the candidate `frozen-source.json` map; its 817 selected hashes match both successful build receipts. The source-bound build excludes that map file from its hashed source-input set, so it is separately bound here.
