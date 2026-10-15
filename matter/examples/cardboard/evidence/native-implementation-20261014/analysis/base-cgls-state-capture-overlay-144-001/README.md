# CGLS selected-state sidecar overlay (144)

This is a source-only diagnostic overlay on frozen source143. It adds the
opt-in `NM_FEM_BASE_CGLS_STATE_CAPTURE=1` sidecar around the existing CGLS
capture. It does not edit the production snapshot, Metal kernels, solver
equations, or acceptance criteria. It has not been built or run on a GPU.

The sidecar is emitted beside the existing CGLS payload as
`<NM_FEM_BASE_CGLS_CAPTURE_PATH>.state.bin` and `.state.json`. The existing
CGLS path and canonical Newton selector remain required; the new variable is
strictly `0` or `1`. The snapshot point is after the Newton correction and
before CGLS direction construction or installation. It is encoded as ordered
read-only buffer blits into a bounded shared staging buffer. The writer uses
exclusive creation, writes and syncs the payload first, then writes the
manifest as the completion record. If capture input buffers, the tangent
cache, the bounded allocation, or a blit encoder are unavailable, the sidecar
is explicitly skipped; capture must not be interpreted as present.

The manifest binds the run, control step, substep, microtick, Newton index,
environment and FEM extents, dispatch metadata, timestep, device and source
fingerprints, and the basename of the companion CGLS capture. The 24-array
table stores accepted/candidate node and tetrahedron records, accepted
material history, the literal candidate material-state buffer, material and
program inputs, Newton correction, base RHS, tangent cache and validity,
FGMRES/solver/status state, and FEM/environment line-search values. The
payload is capped at 96 MiB and also checked against Metal's maximum buffer
length.

Material projection is deliberately not inferred. The tangent-cache builder
uses accepted history and projects into local temporaries while producing
tangent columns. The candidate material-state arena is written by
`nm_fem_validate` later. Therefore `candidate_material_state_buffer` is
retained as a raw buffer with that literal role, and the manifest marks the
current-Newton projected history `unavailable_not_persisted`. A future reader
must adjudicate cache generation and validity against the exact captured
candidate and production reuse rules; the sidecar does not assert a fresh
cache or a material-point qualification.

## Validation status

The frozen-source contract and sidecar-reader suite passes 14 tests. The generated patch applies cleanly in `patch --dry-run` against source143. No host build, Metal compile, GPU run, or native capture was performed; this package is an instrument candidate only. Exact command output is retained in `validation.txt`.

## CPU checks

Run the schema and frozen-source contract suite with:

```sh
python3 -m unittest discover -s tests -v
```

Validate a future native sidecar with:

```sh
python3 tools/read_state_capture.py /path/to/capture.state.json
```

The reader rejects duplicate keys, nonstandard JSON constants, non-adjacent
payload paths, wrong schema/counts, malformed or overlapping extents, wrong
known record widths, and any payload or per-array hash mismatch. It validates
bytes and metadata only; it does not decode record arrays or establish
physical/native qualification. The current tests use a synthetic schema
fixture and source assertions, not a captured native run.

## Source binding

The exact frozen base and generated candidate/patch hashes are in
`overlay-manifest.json`. `tools/make_overlay.py` refuses to run if the frozen
source hash changes. `generation-report.json` records the generated overlay
manifest at the current revision. The candidate contains only
`matter/src/runtime.mm`; no source143 files were modified.
