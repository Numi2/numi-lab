# Verified Git-free sparse source package

The final source archive is bound to commit
`a182448ec9d987ef9446fea384560d61cb5ed4ee` and has SHA-256
`d1b6ead785962f0c0061f15e6d4d80b32ccdbca8b08a1191077ff2951bcc62c1`.
It includes 1,976 selected source files, qualification fixtures and a Git
commit/tree inclusion proof containing 134 required trees. It is a sparse
export, not the complete repository. The previous `9e25d8ca` proof candidate
is retained separately and was not used for this final build.

From a new extraction with no `.git` directory, configure, compilation,
installation, three CTests, source verification, stamping and installed audit
all passed under a macOS sandbox denying network access. Source proof checks
all exported source paths and Git blob IDs and executable-bit modes against the
independently supplied commit; archive SHA-256 and per-file SHA-256/mode are
also checked. The installed marker is `verified-source:COMMIT`, and the
package contains its archive and a receipt with archive/manifest/proof digests
and selected workflow artifact hashes and modes. Neuron consumes the commit
only after a fresh audit with independently supplied expected commit and
archive hashes; otherwise it reports unknown, except explicit user overrides.

An independent source-tree verifier audited the actual installed package both
before and after the native workflow. An extra source file caused stamping to
fail without altering the existing receipt or revision; an installed trainer
mode mutation caused audit to fail, and restoring it restored the audit. The
initial added coverage test failed due to a fixture directory already existing;
the corrected test passed and both logs are retained. An earlier portable
archive run had a text/JSON wrapper failure, retained in the separate
`portable-offline-source-20261108` evidence.

The network-denied native non-G1 synthetic WorldPack smoke ran on M6 at
00:38:37.971–00:38:43.086 UTC on 2026-11-08: one environment, eight steps,
one update, matched selection, pinned inputs unchanged, incumbent retained,
no remaining lock or worker. The unsupported non-humanoid balance screen
remained unknown and refused auto-load, including changed/missing world cases.
This is not physical, learning, UI, replay or performance qualification.

The externally supplied expected hashes identify bytes; they do not establish
trusted authorship. The proof covers the selected source, not a full checkout,
and installed hashes do not attest that binaries were built from that source.
A verifier cannot defend against its own malicious replacement; independent
verification requires a separately trusted copy. Verification and installation
are not atomic against concurrent edits. External OS, system libraries,
toolchain, Python and MLX dependencies remain outside complete identity.
