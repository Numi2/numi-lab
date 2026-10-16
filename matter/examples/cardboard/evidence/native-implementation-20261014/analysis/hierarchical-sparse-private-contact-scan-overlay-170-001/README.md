# Hierarchical sparse private-contact scan overlay

This isolated overlay starts from `source-warm-cgls-refinement-169`. It replaces the sparse Armijo pair-flag scan with a stable three-pass hierarchy while retaining the old one-threadgroup scan for direct native comparison:

1. One 32-lane group handles each 256-pair block and writes local ranks plus block counts.
2. One 32-lane group per environment converts block counts to exclusive global offsets and latches the same overflow failure.
3. A flat scatter writes candidate IDs and pair-to-slot mappings in canonical pair order.

The sparse candidate cap, the 36 MiB scratch budget, pair ordering, physical operator, and acceptance rules are unchanged. The added offset arena is checked in the shared admission helper, host allocation checks, runtime admission, and the shipping-sized arithmetic fixture. At the recorded one-environment shipping pair count of 6,335,568, the calculated scratch is 36,189,188 bytes, including 98,996 bytes for block offsets; this is below the unchanged 36 MiB limit. The two-environment version is rejected by that same limit.

The pre-existing-failure path resets only the count, preserving partial flags, IDs, and status. A newly detected capacity overflow still scatters the first-capacity canonical IDs before later evaluation observes the failed status. CPU contracts and the standalone native checker cover these semantics. The native differential checker also includes capacity-minus-one, exact-capacity, and capacity-plus-one fixtures, block tails, all/none, alternating, deterministic random, mixed healthy/overflow and prefailed environments, and the full pair-domain fixture.

Validation used C++20 host syntax checks, the CPU-only scan contracts, the existing admission arithmetic check, and strict Metal O0/O3 compile plus metallib link. The O3 dependency file confirms that the candidate `contact.metalinc` was compiled. No native Metal checker was dispatched and no physics transaction was run. The two Metal warnings are from frozen `fgmres.metalinc` variables `sharedValid` and `refinementValid`; this patch does not touch that file. Initial misconfigured C++17 syntax attempts and their logs are retained alongside the corrected passing commands.

Files are regular copies, not symlinks. Apply `source-overlay.patch` with `patch -p1` to the exact base snapshot only after verifying `source-overlay.json`. This candidate is not a frozen source snapshot, not shipping qualification, and remains subject to native old-versus-new differential review by the parent task. Source173 has independent warm-state edits; the runtime hunks here should be merged onto it without replacing those changes.
