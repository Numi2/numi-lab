# C300 compact boundary audit (final local-data pass)

This offline audit checks the exact two-band, three-tetrahedra-per-face volume connectivity emitted for the paired skin/fat layers. It verifies raw-face orientation before the production positive-orientation swap, positive Float32 tet volumes, count-one exposed faces, internal face pairing, the 45 authored open-edge sidewalls, and closed-boundary edge orientation/self-intersection.

Final solve-032c used the corrected donor-local CSV (hash-pinned in the run receipts). Tissue reports only donor-local coordinates and frame-relative deltas changed; Float32 world geometry and thickness fields are unchanged. The audit passes with 2,332 vertices per layer, 4,619 source faces, 27,714 tets, 9,418 exposed boundary faces, 180 rim sidewalls, 0 nonmanifold faces, 0 bad boundary edges, 0 self-crossings, and positive signed volume. The solve-031 negative control retains 4 boundary self-crossings and fails as expected.

This is an offline topology/geometry audit only. It does not qualify runtime contact, whole-body stability, or physical behavior. Candidate topology and layer CSVs are referenced by path and SHA-256 in `manifest.json`; they are supplied by the Tissue bundle and intentionally not duplicated. The prep helper header is also external and hash-pinned.

To rerun, use `replay.sh /absolute/path/to/admitted-layer-coordinates.csv /absolute/path/to/new-output-directory` after verifying the pinned external inputs.
