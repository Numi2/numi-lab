# Typed source namespaces for registered physical skin/fat

The outer, inner and deep surfaces may have different registered source index spaces. A source domain is now pinned by its name, anatomical role, SHA-256 and authoritative vertex/face counts. The same textual domain can name separate inner/deep tables without conflating their IDs. Per-node and boundary maps preserve direct or barycentric source identity. Derived volume interiors explicitly carry no surface ancestry.

The volume importer, physical-volume validator and reduction-label builder validate the same typed metadata. The labels retain a separate internal metadata checksum; it is not a substitute for the pinned source hashes or the physics fingerprint. A role-aware source/body callback must verify source binding. Legacy single-source callers keep their existing validation.

The file-level helper reads the existing pinned TetGen files, combined-node CSV and facet-ancestry CSV transactionally. It builds exact O/I/D facet maps and accepts only retained parent-backed head walls. It does not admit a physical volume, alter mass equations or install geometry.

Seven existing/narrow check binaries passed on SSH Numi, covering mixed-domain positive cases, bad role/hash/bounds/interior metadata, failure rollback, six-tet facet ownership, mass mapping and legacy behavior. The full native target built. Its source-only preflight emitted zero physics steps, created no runtime, and reproduced all nine baseline geometry/pose/parent/mass outputs byte for byte. Exact commands, hashes and outputs are in checks.json.

Production layer geometry, conservative mass transfer, native owner replacement and accepted motion qualification remain pending. Fixture passes and source-only parity do not qualify the replacement human.
