# Erratum for the retained v3 344 report

`validator-v3-attempt-states/reports/capture344-admitted-v3.json` is preserved byte-for-byte. Its status says the admitted restart capture was inconsistent because its gate `q_scale_and_P_orthogonality_relative = 0.0444142` compared Euclidean `Pᵀq` against `0.002`.

That gate is mathematically inapplicable to the source-101 restart projection. The kernel solves `(PᵀAP)c = PᵀAq` and updates `q <- q - Pc`, `Aq <- Aq - APc`; the applicable orthogonality is `PᵀAq ≈ 0`, with nonsymmetric `A`. The corrected v4 report measures `PᵀAq` from captured and freshly recomputed `Aq`, and keeps `Pᵀq` as a non-gating diagnostic. The old v3 failure must not be cited as evidence that the 344 augmentation was rejected by the native solver; the native run itself still accepted zero physical steps.
