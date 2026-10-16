# Publication037 finalizer extension

This package adds an append-only archive finalizer after the first critical23 extension attempt. The earlier extension successfully copied and verified its five case payloads, then failed in its last path-list check because the check compared evidence-relative paths with checkout-relative Git paths. No data payload was lost. That failed script and a preflight hash-constant error are retained in publication037-extension-source-001; the current package retains the second-run failure receipt.

This finalizer starts from the already-updated 1,931-record summary, appends this package as one additional case, and computes the repository allowlist by prefixing archive paths with the evidence root. It preserves the 1,879-record historical prefix and the earlier 52 publication037 records exactly. It includes ignored payloads only in the explicit allowlist; it does not stage or commit anything. Active native545 remains excluded.

The critical23 evidence is a bounded material replay: 138 valid projection/stress/tangent rows over 23 selected tet/basis pairs, zero physical FEM transactions, tangent-vs-strict-FP64 maximum relative error below 1e-3. Twelve y-sign disagreements are smooth-law indicators. Native finite-difference comparisons remain unqualified; the reported endpoint-signal counts are descriptive and do not introduce a 1e-6 native-FD gate.
