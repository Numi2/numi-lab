# Repair and transaction boundary

The preregistered baseline shows environment zero returning to phase 0.02/action 0.005 after growth; the preallocated control correctly reaches 0.04/0.010, with or without a host abort. Retain that result.

Allocate all replacement buffers before changing the active set. Encode zero-initialization followed by a copy of the existing phase and previous-action prefixes on the borrowed command buffer. Keep the old complete allocation set and the growth command until a subsequent owner call establishes whether that command was committed. If a later call sees a different, still-uncommitted growth command, restore the old allocation before retry; if it was committed, retire the old set. This supports the existing serialized, same-owner command contract and does not claim overlapping submissions or partially executed GPU-error recovery. Do not expose or commit a second queue, wait or CPU readback.

Qualify the exact preregistered growing/preallocated and abort controls on Apple Metal, then the existing physical-rejection and capacity regressions. Preserve any build or test failure. The test masks are diagnostic across distinct residents and do not establish identity migration, simulator reset behavior or learning performance.
