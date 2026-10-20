# Replay of 041b root21/tick3 failure

This diagnostic replay requested 22 steps at 125 µs and reproduced the default32 failure after 21 accepted roots (2.625 ms): root 21, tick 3, Matter status 6, object 19, local failing index 2623, three microsteps, diagnostics `[3.7129670005242588e-08, -4.6978958707412106e-15, 1, 0]`. It exited 1 after 356.176031 s.

The complete coupled CSV, common failure receipt, and surface audit are byte-identical to `../native-integrated-041b`; code-source and executable/library hashes match as well. Replay was invoked with 22 requested steps to stop at the first failure witness. Its CSV contains 21 accepted roots. Compact stdout/stderr selections retain fascia admission, ownership/mass, and terminal failure markers. Raw logs, the pack, and full 12-trial payload are not copied; hashes and original paths are recorded in `raw-log-identities.json`. The analysis owner may retain the selected trace JSON and full trial payload in the separate `analysis/` directory.

The separate `verification-041b` checks for 26 source regions, geometry/mass, 12 compiled objects and offsets, and late-reject/retry remain passed. The 32-step prediction remains refuted by the 21-root failure; this replay only reproduces that exact failure and does not support a stability improvement. No sustained-support or physiology claim follows.
