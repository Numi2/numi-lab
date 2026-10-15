# Source101 cached-versus-fresh operator replay

This package prepares a read-only, no-transaction native replay of the
`source-boundary-reset-order-101` capture from Newton 23, control step 2. The
checker invokes production `nm_fem_apply_operator_elements` on the exact same
captured candidate, accepted material state, per-environment parameters,
captured 9x9/tet tangent cache, and correction vector with `useCache=1` and
`useCache=0`; each result is then passed through production
`nm_fgmres_gather_nodes`. It compares both actions with the fresh action already
captured by the source101 diagnostic. It does not advance or commit a Matter
transaction.

The captured correction is unusually large, but it is explicitly labeled by
the capture as `delta_v` in m/s, and the corresponding operator convention is
`A*delta_v=M*delta_v-dt^2*K(delta_v)`. Environment 0 has correction L2
`24346.271154909504 m/s`, maximum nodal magnitude `13760.171400398754 m/s` at
node 479 (`[-160.436874, 8.55699348, -13759.2334]`). These values are retained
verbatim from the captured linear solve and are not rescaled or treated as a
physical velocity; the capture source marks them as the FGMRES correction
unknown. Environment 1 is the quiet reference with zero correction. The
capture reports the residual convention and units in its header, and the
payload records the exact values.

The native checker is compiled against the source101 headers and provenance
libraries, with the exact source101 run metallib. Compilation succeeded with
`-Wall -Wextra -Wpedantic -Werror`. The executable has **not** been run yet;
that would dispatch Metal kernels and is waiting for the parent task's GPU
slot. No native comparison results are claimed here.

## Files

- `prepare_replay.py` repacks and binds the exact source101 capture and compiled
  package. It refuses mismatched source, capture selector, package, or shader
  hashes.
- `prepared/replay-input.bin` is the 2,054,152-byte binary replay input.
- `prepared/input-manifest.json` and `.seal.json` bind its source, capture,
  compiled package, metallib, and payload hashes.
- `cached_operator_replay.mm` is the standalone production-kernel checker.
- `build/cached_operator_replay` is its CPU-compiled arm64 executable.
- `prepared-v1/` preserves the first payload draft, before the independent
  captured fresh-action vector was appended. Use `prepared/` for any replay.

## Reproduce CPU preparation and compilation

The payload is already prepared; `prepare_replay.py` intentionally creates a
new output directory and will refuse to overwrite it. To reproduce into a
separate location, copy the script and change only its `OUTPUT` path, then
verify that the resulting payload and manifest hashes match the sealed values.

The exact compile command is recorded in `package-manifest.json`. Running the
compiled program takes three arguments: the source-bound compiled package,
the prepared binary payload, and a new output directory. It uses Metal and
must only be run when the parent assigns the GPU slot.

## Scope limits

The compiled world has two environments, one FEM object, 882 nodes and 1,920
tetrahedra per environment, with zero rigid proxies, rigid generalized rows,
contact pairs, fields, or cohesive faces. The replay checks the complete
mechanical FEM operator slice for this capture, not the coupled contact/rigid
operator. Newton 23 was rejected. This is a diagnostic replay of its captured
linearization, not an accepted physical state or solver-convergence claim.
No noise floor is inferred from same-run repeated outputs.
