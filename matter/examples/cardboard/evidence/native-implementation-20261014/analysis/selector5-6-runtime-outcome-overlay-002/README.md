# Selector 5/6 complete runtime outcome diagnostic overlay

This package is a fresh checker-only candidate against frozen source 129. It runs selector 5 and selector 6 in separate runtime instances even if the first fails, captures both `RuntimeStateSnapshot` results, and prints both complete outcomes before applying the original status, certificate, nonzero-correction, fixed-node, and representable-motion gates.

The output includes exact raw bytes and element counts for all 59 vectors and all scalar fields in the source-129 `RuntimeStateSnapshot` ABI, plus decoded per-environment status and certificate values, fingerprints, transaction cursor, resident-byte estimate, and initialization/encode/command errors. A snapshot unavailable because setup or encoding threw is explicitly marked unavailable; the diagnostic does not synthesize state. This provides the evidence needed to distinguish a runtime-selector difference from a fixture or setup failure.

The production runtime, Metal kernels, physical inputs, selectors, force, and gates are unchanged. Validation is CPU-only: frozen-base contract checks, Objective-C++ `-Werror` syntax validation, and patch dry-run. No native GPU run was performed.

## Package 001 integrity incident

Package 001's original sealed manifest and patch are preserved unchanged. A post-seal edit changed its candidate bytes from the manifest-bound SHA-256 `2727109d414ed75662d2f135002d4b04bfa6ed1a095459008362c8a7ab5a065f` to `4d1aca6a92f2bb28c0c3aa15f37e32bf531dfb091d71cef3792c2ecc9a375d9e`. Those changed bytes are preserved in `post-seal-001-candidate-preserved.mm`. Root's integration preflight caught the mismatch before cloning or patching source 131. Package 001 was then restored from its original patch against the frozen 129 base, and its complete manifest inventory now verifies. Package 002 is separate; it does not revise or replace package 001 evidence.
