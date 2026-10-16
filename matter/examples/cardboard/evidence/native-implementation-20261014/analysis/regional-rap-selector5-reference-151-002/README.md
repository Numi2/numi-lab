# Selector-5 graph/RAP CPU reference, source 151

This package implements an FP64 algebraic reference for the source-151
selector-5 graph map and the unaugmented base RAP correction. It is a bounded
reference and input-contract tool, not a native Metal test.

The graph map is `H = G D E`. `E` gathers only free candidate-node entries;
`D` applies one original-patch-matrix residual refinement using the captured
pivoted local factor; `G` scatters patch corrections and divides by exact node
coverage. Its transpose reverses these operations, including coverage scaling
before patch solves. The base selector map is
`S = H + (P - H AP) R^-1 P^T`, with transpose
`S^T = H^T + P R^-T (P^T - (AP)^T H^T)`. The local refinement transpose uses
`D^T = 2 F^T - F^T A_p^T F^T` in exact arithmetic, where `F` is the captured
LU solve map. All reference dot products use `math.fsum`.

The loader delegates payload/dtype/hash parsing to the source-151 capture
reader, independently checks exact compiled topology, free-mask, and
potential-mask bytes supplied by the caller, validates graph layout FNV-1a,
and checks the candidate fixed tags against the compiled free mask. It
supports only the current all-nodes-potential policy. A caller cannot label
the captured base view as the map actually consumed by an arbitrary FGMRES
callback. Restart augmentation and callback consumption are not identified
by this sidecar, and no arbitrary-vector native output is captured. The
actual selected/augmented view is therefore explicitly unsupported here.

The test fixture has three nodes, two overlapping patches, nonsymmetric local
matrices, a pivoted local factor, a pivoted base RAP factor, a fixed node,
coverage averaging, completed-tail behavior, and invalid-factor controls.
Tests exercise transpose dot-product identities, factor and RAP
reconstruction, fixed masks, malformed coverage/CSR, nonfinite input,
completed-tail zeroing, and fail-closed augmented-view claims. They do not
exercise the strict binary loader against a real selector-5 native capture;
the owner has not produced such a capture yet.

## Reproduction

From this directory, run:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v test_selector5_graph_reference.py
PYTHONDONTWRITEBYTECODE=1 /Users/home/.pyenv/versions/3.10.12/bin/python3 run_synthetic_reference.py
PYTHONDONTWRITEBYTECODE=1 /Users/home/.pyenv/versions/3.10.12/bin/python3 audit_source_contract.py
```

All three are CPU-only. The source-contract tool checks source file hashes
and extracts the specific kernel bodies and source anchors listed in
`source-contract.json`. `manifest.json` binds the scripts, copied reader and
reference, and generated reports. No production source is modified.

## Limits

- The reference uses binary64 arithmetic and the exact algebraic transpose;
  it does not reproduce FP32 rounding or native thread ordering.
- Captured base factors are input-specific. The code does not reconstruct an
  arbitrary current A or claim patch factors are fresh unless the caller also
  supplies and identity-binds that same-generation A.
- Selector-5 native apply/transpose parity, actual selected restart-cycle
  consumption, solver convergence, physical contact behavior, and shipping
  qualification are not established by this package.
