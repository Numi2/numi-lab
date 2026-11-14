# Initial long-flight invocation errors

The first straight and turn invocations used tag `longflight`. The existing runner selects the required candidate policy packs only for tags beginning `rebound`, so the commands instead selected incompatible older packs. Both exited 1 during policy contract compilation before stepping the simulation. Their commands, input hashes and complete native stderr are retained under the `*-20261126-longflight` directories.

An old remote queue remained alive after its local SSH session was interrupted. A later queue reattempted the same paths, which correctly refused to overwrite existing directories and logged FileExistsError; those duplicate-runner errors are retained. No physical result can be inferred from either invocation. The corrected run uses tag `rebound-longflight`, which selects the existing compatible policy packs, and writes to fresh paths. Its scientific plan and controller binaries are unchanged.
