# VTCCD exact-capture absolute payload path repair 001

Base snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125`.

Native 414 stopped in the reader before any Metal dispatch because both retained source-122 trial manifests store `payload.file` as an absolute path to the binary beside that manifest. The checker required a basename. This overlay keeps both exact manifest, payload, and 336-byte observer SHA checks, the expected source/device fingerprints, exact call identity/return bits, and the budget-16 bitwise replay gate. It accepts absolute payload paths only when symlink-resolved parent directory equals the manifest's symlink-resolved parent. Relative inputs remain restricted to basenames and are resolved beside the manifest.

A CPU-only `--validate-inputs-only <trial21.json> <trial22.json>` mode executes the same loader and all exact retained input checks, verifies the expected uncertified-miss and certified-hit words, and returns before `MTLCreateSystemDefaultDevice`. It is not a GPU replay or CCD qualification. `validate_inputs.py` also feeds a modified cross-directory absolute path and confirms the sibling-directory guard rejects it before payload admission.

The normal replay path is otherwise unchanged. The exact 16-step replay, higher-budget outcomes, and the original 405/406 fixture failures remain to be evaluated by a later GPU run; this package makes no such result claim. The source125 checker, native 414 artifact, exact captures, and sealed `private-vtccd-budget-replay-overlay-001` are unchanged.
