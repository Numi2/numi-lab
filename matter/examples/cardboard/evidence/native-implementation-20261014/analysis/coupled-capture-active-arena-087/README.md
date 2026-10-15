# Coupled Newton contact-arena capture overlay 087

This external diagnostic overlay is based on the frozen 086 source snapshot:
`/Users/home/cardboard-implementation-20261014/snapshots/source-captured-point-and-contact-scope-086`.
It changes only `matter/src/runtime.mm` in that snapshot. It does not change solver mechanics, production arenas, acceptance, or the 32 MiB capture cap.

The prior capture scope rejected `deformableContactCapacity > 16384` before considering the actual readback size. The 286 run therefore skipped at root 1201 / Newton 12 with `deformable_contact_capacity_out_of_scope`; its physical failure and skip receipt remain evidence for source 086. In 087, that arbitrary diagnostic-only capacity cutoff is removed. Full arena buffers remain mandatory and are copied only if every typed extent is present and the aggregate payload is within the existing 32 MiB cap. The skip receipt reports the observed dispatch dimensions, capacity, each relevant buffer length, the fixed cap, a saturating estimate for the complete contact arena, and bytes requested from present sources.

For the authored two-layer test fixture, 48 tetrahedra expose 192 face slots. The conservative unordered candidate reservation is `192*191/2 = 18,336`. Per slot, the full readback contains a 432-byte contact, 16-byte candidate, two 48-byte histories, active/global indices, and six-node incidence: 576 bytes per slot. Environment records (typed 16-byte sentinel counts/offsets, 36 node ranges, one dispatch record) add 640 bytes. The minimum full contact-arena readback is therefore 10,562,176 bytes. Other captured state shares the same 32 MiB payload cap, so 087 still skips if the complete aggregate does not fit.

After the GPU-to-shared snapshot completes, the callback checks that each environment's active count is within capacity and that every active-prefix index is in range, unique, and points to a record whose valid bit is set. It does not require the total number of valid records in the full arena to equal the active count: unused valid records are captured and reported explicitly. The full capacity and active prefix are both present, so no truncation is hidden. A failed prefix proof emits a failure record and no replay payload.

`coupled_contact_coverage.py` verifies payload and array hashes, contiguous full payload coverage, required full-size contact/candidate/history/incidence arrays, active-count-buffer agreement, prefix range/uniqueness/valid bits, and the reported count of unused valid records. Its unit tests include truncated arenas, truncated active prefixes, duplicate indices, metadata disagreement, active-count buffer mismatch, and unused valid full-arena records.

## Verification

Run:

```sh
python3 verify_overlay.py
python3 test_coupled_contact_coverage.py
```

The Objective-C++ runtime syntax check passed against the frozen 086 headers with `clang++ -fsyntax-only -Wall -Wextra -Wpedantic`. No GPU run has been performed for 087. The eventual root-owned native attempt must bind the exact 087 runtime hash, preserve the source-086 failure, and compare diagnostic-on/off histories before interpreting any captured replay.
