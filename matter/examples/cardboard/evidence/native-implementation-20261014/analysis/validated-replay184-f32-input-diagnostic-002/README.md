# Run 184 JSON-to-FP32 input round-trip audit

This is a CPU-only replay of the 2,880 paper material-center records in run 184. It uses the retained compiled residual bytecode, the captured FP32 parameter/immediate bit patterns, and captured state words. It compares two input interpretations while holding those compiled programs and state inputs fixed:

1. JSON numeric tokens parsed as binary64 values, as the existing replay path does.
2. Each serialized high/low deformation and rate component rounded back to IEEE-754 binary32 and then widened exactly to binary64, matching the native material input representation.

The independent Python bytecode interpreter reproduces every stored raw-decimal compiled-bytecode residual row exactly (maximum absolute difference 0 across the replay). On the target medium tet 1291, environment 0, row 10, raw-decimal replay is `1.0147715764095697e-4`, while the binary32-promoted replay is `-6.806494752360426e-11`; the captured native accurate residual high+low is 0. Across 2,880 centers, the raw-decimal/native absolute-error median is `1.2774732375988966e-12`, p95 `4.002202218110181e-5`, max `1.0147715764095697e-4`. With binary32 promotion, median is 0, p95 `3.6645787628327946e-11`, max `8.378640617063571e-11`.

The raw decimals are the text serialization of native single-precision inputs; the discrepancy arises when the FP64 oracle interprets those decimal spellings as more precise values instead of round-tripping them through FP32. Target F-high changes include `0.606021821` → exact FP32-promoted `0.6060218214988708`. F-low is zero in the target record. No accurate residual program contains a RATE opcode, so changing the rate representation has no effect in this replay.

The run's non-timeout exit 2 and zero physical steps are retained. This identifies an oracle-input representation mismatch; it does not validate the material model, native solver convergence, or physical behavior.

## Reproduce

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/validated-replay184-f32-input-diagnostic-002/replay184_input_roundtrip.py
```

The script validates the raw capture SHA against both replay header and summary, checks the run receipt/build/external manifests, verifies F/F-low/rate against the parent tetrahedron records, and writes `report.json`. Relevant input bindings are recorded in that report. Current script SHA-256:

`96772903c85a9973eb92b3a138832040f4185010460231829948b8a2cd97f983`

Current report SHA-256:

`077710a6bcf64803d3c1e1d140242e86295b06e9bad708355094cf295cafa6b7`
