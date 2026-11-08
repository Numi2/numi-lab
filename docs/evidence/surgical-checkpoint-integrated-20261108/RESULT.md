# Integrated surgical checkpoint integrity verification

After integrating the surgical checkpoint integrity change, source revision `531b2ef7f` was rebuilt on `ssh numi` (Apple M6). The native tissue-rest setup, checkpoint restore, corruption rejection and restore/rewind/hold tests passed 4/4; see [ctest.log](ctest.log) and [SHA256SUMS](SHA256SUMS). This independently checks the integrated source and binary; it does not relabel the original source-pinned measurements.

The fixture is synthetic tissue-rest, and the hold path cold-initializes MetalWorld contact caches. This is not a complete resident-owner restore, physical-force qualification, or a surgical continuation. Both experimental CCD and AL options remain disabled.
