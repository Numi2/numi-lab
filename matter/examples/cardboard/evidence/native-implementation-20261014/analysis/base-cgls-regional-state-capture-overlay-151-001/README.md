# CGLS regional state sidecar overlay 151

This is an isolated, default-off diagnostic overlay prepared against the sealed
source-149 snapshot. It adds a second sidecar for a selected CGLS Newton state;
it does not change the FEM operator, solver acceptance, or production history.
No native build or GPU run was performed for this overlay.

The source base is
`/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-block-right-preconditioned-149`.
Its runtime SHA-256 is
`f859b1ee45390dd844fcbd883356b931ab71d2c36260f53a4b03600b04fb3959`;
the source snapshot's `frozen-source.json` SHA-256 is
`f98cbfc0aac642d0578d7f0efefb19c288632b73f7b5607b70e19a15e7963bb5`.
The bound source-149 overlay manifest is
`analysis/native-cgls-block-right-preconditioning-overlay-149-002/source-overlay.json`
with SHA-256
`e158080398d47e1974e34ffcd71af5cd7e563d0fd08c4eb4e53482e7f9a94906`.
The verified source-149 patch SHA is
`e23ed54a067f2f2ad822d84172d21df225aa112e28307f0cbe6299a175da8a86`.

The 151 work began with source 145 plus the owned diagnostic additions. It was
rebased by a three-way merge using exact 145, 151-work, and 149 runtime files.
The merge had no conflicts. The resulting candidate is
`candidate/matter/src/runtime.mm`, SHA-256
`1c0d2fd0d376ecc5e7ea9007ab6cc67a77fbd19bbba9f3b22fcbeca4c5c06a08`.
The source-149-relative patch is `patch/runtime.patch`; applying it to the
source-149 runtime reproduces the candidate byte-for-byte. The complete file
inventory and hashes are in `overlay-manifest.json`.

## Capture behavior

Set `NM_FEM_BASE_CGLS_STATE_CAPTURE_REGIONAL=1` together with the existing
`NM_FEM_BASE_CGLS_STATE_CAPTURE=1` selected-root/Newton capture. The existing
selected CGLS capture path and admitted CGLS scratch must be enabled, the
regional Galerkin preconditioner must be selected, and the existing bounds
must admit the state. Unsupported combinations and overlapping bounded
readback observers are rejected before the capture is encoded. The default
when the new variable is absent or `0` remains unchanged.

The established state pair remains `<capture-base>.state.bin/.state.json`.
The new pair is `<capture-base>.regional.bin/.regional.json`. The regional
manifest binds the state-payload SHA and repeats control/substep/microtick,
micro-generation, Newton, source/world/program fingerprints, device identity,
and `dt`. The existing state manifest gains the additive `micro_generation`
identity field so the reader can prove that join. The sidecars are blitted
from the same command buffer and written only after it completes. Files use
exclusive creation; a failed write removes partial outputs and reports a
skip. No array is silently truncated.

The regional payload records authored `P`, per-environment effective `P` and
`AP`, coarse RAP LU/pivots/validity/RHS/coefficients, and restart augmentation
raw buffers when enabled. For selectors 2–4 it also captures the exact nodal
fine diagonal and node incidence ranges; selectors 3–4 include all three inverse
block rows, preserving row0.w validity/fallback markers. Candidate node records
retain the raw fixed-node tag, and the joined state sidecar carries FGMRES health
and Armijo metrics. Selector-5 captures also include immutable graph patch
descriptors, membership, row map, CSR, coverage, original matrices, LU, pivots,
and factor-valid words. Every array is little-endian, typed, 16-byte aligned,
and SHA-bound. Raw augmentation is admissible only when its generation words
and validity flags match the selected root/substep/Newton. The captured
candidate material-history buffer is explicitly marked as not proven to be this
Newton's projected material state. The capture has no native output for an
arbitrary regional-apply vector: saved coarse RHS/coefficients describe only
the last recorded coarse input.

The aggregate readback budget remains 96 MiB and the regional table is capped
at 32 arrays. The budget includes the established state snapshot, regional
arrays, existing CGLS scratch snapshot, trace, and optional right-scale rows.
Each size uses checked arithmetic and device-buffer-limit checks. This is a
readback cap, not a runtime solver or shipping admission claim.

## Reader and checks

`candidate/tools/read_regional_capture.py` is a standard-library reader. It
rejects nonstandard JSON tokens, duplicate keys, symlink or nonadjacent
payloads, oversized or mismatched payloads, noncanonical array padding,
overlaps, bad hashes, unjoined source/device/time/generation identities,
missing exact state or selector-specific fine-smoother arrays, invalid record
widths, and graph CSR/row/pivot inconsistencies. It reports base pivot and
fine-smoother numeric admissibility separately from successful byte parsing. It reports
that the capture is diagnostic only and never claims physical qualification.

The synthetic parser contract suite is
`candidate/tools/test_read_regional_capture.py`. Run it with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 candidate/tools/test_read_regional_capture.py
```

Result: 17 tests passed. These parser fixtures are synthetic and do not count
as native evidence. `validation/python-contract-tests.log` retains the exact
test output. Python source compilation via `compile()` also passed without
writing bytecode; it checks syntax only. No ObjC++ syntax, Metal compilation,
CMake build, or GPU execution was run. Source 149 and all frozen snapshots
remain unmodified.
