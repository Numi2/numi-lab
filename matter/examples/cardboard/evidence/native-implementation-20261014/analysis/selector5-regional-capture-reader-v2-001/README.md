# Selector-5 capture reader v2: producer topology digests

This separate v2 reader consumes the `source_topology_sha256`, `source_free_mask_sha256`, and `source_topology_digest_encoding` fields added by the RAP-only topology overlay. The sealed reader v1 remains unchanged.

The producer hashes canonical bytes, not the compressed NPZ file: little-endian `u32 node_count`, little-endian `u32 tetrahedron_count`, then the full ordered tet connectivity as row-major little-endian `u32` (four node IDs per tet). The free-mask hash is over exactly `node_count` raw bytes with `1=free` and `0=fixed`. The reader derives the free mask from the selected candidate state (`restAndFixed.w`), requires each tag to be exactly `0.0` or `1.0` as the native graph builder does, checks both producer digests, then rebuilds the deterministic graph from the supplied NPZ `nodes` connectivity and compares every global-ID patch, row, and node-patch map entry. It also requires NPZ `node_count` to be a scalar integer. Same-sized meshes with different connectivity bytes are rejected even if adjacency or patch counts happen to match.

Use:

```sh
python3 read_selector5_capture.py <capture.regional.json> <capture.regional.bin> \
  --topology-npz <same-run-normalized-topology.npz> --require-topology \
  --output <validated-summary.json>
python3 test_reader.py
```

Without a topology file, the reader checks that the candidate-derived mask matches the producer's `source_free_mask_sha256`, but returns `source_hash_present_topology_input_missing`; it does not certify topology/global-ID membership. Strict mode fails without the NPZ. The NPZ should be a retained normalized artifact from the same compiled world/capture, not a mesh selected later by matching node and tet counts.

As an encoding cross-check, applying the exact producer byte recipe to the source-182 audit topology NPZ yields `source_topology_sha256=52d495969f0220838a87b2dd7984961265fc0e6b1891f2d3e157b2ef17ae7511`; the candidate env0 mask yields `source_free_mask_sha256=dae5a5047292398f5a4ebfae0f961daa5a76ec53af59bab106f326de08aeefd8`. Source 182 itself is selector 3 and its current manifest has no selector-5 topology fields, so these are encoding checks only, not a validated selector-5 capture.

Seven CPU interface tests pass, including exact producer encoding, a positive global-ID reconstruction, rejection of same-count alternative connectivity, and rejection of altered captured IDs. No GPU run is part of this package.
