# Shipping checkpoint physical-arena correction (isolated source171 overlay)

This overlay fixes checkpoint shape validation for buffers that Runtime snapshots as full physical arenas and Runtime::restore requires byte-exact. `checkedBytes(count, sizeof(T))` allocates `max(count*sizeof(T), sizeof(T), 16)` bytes, so empty and short logical arrays can carry typed placeholder records. The prior shape check compared those snapshots to logical counts and rejected a zero-face coupon at `cohesiveFaces` even though its snapshot/restore arena was complete.

The candidate adds checked physical-record sizing and applies it only to full-arena fields. Logical `readCount`/bounded arenas remain exact logical lengths, and Runtime::restore remains the final device-side exact-layout check. No operator, accepted-state content, tolerance, or runtime allocation changed.

## Validation

- Focused C++ test: passed 6 checks, compiled with the frozen169 include tree plus candidate tool sources using `-O3 -std=c++2b -Wall -Wextra -Wpedantic -Werror -UNDEBUG`.
- Patch dry-run: `patch --dry-run -p1` against source169 passed for both modified files.
- Whitespace check: `git diff --no-index --check` on both base/candidate pairs passed.
- No full build or GPU execution was run. Native 470 remains a failed checkpoint attempt; this overlay is a diagnostic fix candidate, not a successful save/restore qualification.

See `overlay.patch`, `validation/`, and `manifest.json` for exact bindings.
