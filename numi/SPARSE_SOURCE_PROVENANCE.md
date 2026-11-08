# Sparse source bundle provenance

`python/metalrobo/numi_source_provenance.py` creates an offline Git inclusion proof for
an already prepared source bundle. A bundle contains `source/`, a
`BUNDLE-MANIFEST.json` with revision and per-file SHA-256, mode and size, and
optionally `qualification/` files listed in that manifest. `create-proof`
reads the repository commit and required tree objects, storing them as
`GIT-PROOF.json`. It supports SHA-1 Git repositories and regular files.

After archiving the bundle, obtain the expected archive SHA-256 and expected
commit through an independent channel. With an extracted source directory
and a separately built/install prefix, run:

```
python3 python/metalrobo/numi_source_provenance.py create-proof --repo REPOSITORY --bundle BUNDLE
# archive BUNDLE including GIT-PROOF.json, then obtain independent expected hashes
python3 python/metalrobo/numi_source_provenance.py stamp --archive ARCHIVE --source EXTRACTED_SOURCE --prefix PREFIX --expected-revision COMMIT --expected-archive-sha256 SHA256
PREFIX/libexec/numi/numi_source_provenance.py audit --prefix PREFIX --expected-revision COMMIT --expected-archive-sha256 SHA256
```

The verifier checks the complete exported source path set, file content,
manifest lengths and modes, and Git commit/tree/blob hashes. It verifies that
the source used for stamping equals the archive and rejects extras, missing
files, symlinks and changed modes. The installed archive and receipt carry the
verified source identity, manifest/archive digests and selected installed
artifact hashes and modes. Verification happens before the revision marker is written.
This validates identity of the sparse selection, not completeness of a whole
checkout, trusted authorship, or a claim that binaries were built from the
verified source; the build process and other dependencies need their own
records. The expected hashes must not be taken solely from the archive itself.
For independent verification, run an independently obtained trusted copy of
the verifier; the installed verifier cannot defend against its own malicious
replacement. Audit is not atomic against concurrent installation changes.

A stamped install's `REVISION` is marked `verified-source:COMMIT`. Its Neuron
command reports the commit only if `NUMI_SOURCE_EXPECTED_REVISION` and
`NUMI_SOURCE_EXPECTED_ARCHIVE_SHA256` are provided and a fresh installed
`audit` succeeds; otherwise it reports `unknown`. Explicit existing Neuron
revision overrides continue to work. This does not bind all system libraries,
Python packages, environment variables or concurrent changes during a build.
