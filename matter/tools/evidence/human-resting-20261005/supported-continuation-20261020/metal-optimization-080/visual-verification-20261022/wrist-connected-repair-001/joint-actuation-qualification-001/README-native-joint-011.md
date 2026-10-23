# Initial joint construction and matched tendon controls

The native probe now accepts unique non-root initial joint coordinates after the existing fresh-reference posture flag. Existing source ranges, NHEQ projection and fresh selected-tendon owner admission remain mandatory. These coordinates initialize free dynamics; they do not hold or animate the joints.

The zero-activation Achilles matched control now requires force/moment closure but no nonzero activation increment. Loaded trials still require a nonzero increment. No physical equation or closure tolerance changed.

The exact-projection knee auxiliary envelope is prepared by numilab-human main a12836cf5ddd1015cecdaf1e116669dde0cd3eb2. This explicitly selected compatibility policy expands only dependent auxiliary hard ranges around their exact polynomial image; primary knee ranges, polynomial equations, defaults, masses and attachments remain unchanged. It is not parity with the source simulator's soft limit dynamics.

## Validation

Clean base a1a4b5340181fa48caf3006b5a1b8c565d2cc471 plus the two native changes compiled with warnings-as-errors, linked against the retained profile-build-039 runtime, and ran three matched baseline/loaded pairs. The source SHA256 is 3e5c0963ecde905ce9e2377b7fcee80cf5f7d51d2b4d22ca38073cd4d63aa2c5; binary SHA256 e2eec419dd8c8f09c0e8d6017eecc188587a1b2cd30749e19e9c6fe773183195. Each condition accepted 8 steps at 125 microseconds and reproduced bitwise: 48/48 total accepted steps. Knee pairs used equal initial-only q106=q120=0.05 rad and exhibited opposing bilateral flexor/extensor responses. Negative duplicate/root-option controls were rejected. Zero-drive and 0.5-drive Achilles pairs passed tendon force/moment closure.

Clean pairs are in results/011-clean-*. Source/asset/binary/library identities and exact commands are retained in build, invocation, execution, pair and assessment receipts. The original dirty-source trials, parser failures and failed earlier trials remain on SSH Numi and are not substituted for these clean-source results. Assessor v4 fixes literal escaped-newline emission; it changes no physical gate.

## Limits

These are 1 ms selected-tendon articulated core tests, not whole-body skin/fat contact, natural range of motion, gait or physiological qualification. The instantaneous 0.5 Achilles activation produces large ankle velocities; a pass establishes response/closure/replay only. The broader joint skin/fat candidate remains unadmitted while deep-fat self-intersections are repaired. There is no whole-body performance claim.
