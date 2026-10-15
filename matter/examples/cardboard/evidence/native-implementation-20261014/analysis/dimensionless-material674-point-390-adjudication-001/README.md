# Native material point 390 adjudication

This package independently decodes the source-bound native diagnostic at environment 0, tetrahedron 674. It checks the original local residual and scaled-correction gates, the trace-recorded dimensionless line-search merit, the stored row weights, and one FP64 stress/tangent action comparison.

Run with Python 3.10 and NumPy available:

```sh
python3.10 /Users/home/cardboard-implementation-20261014/analysis/dimensionless-material674-point-390-adjudication-001/adjudicate.py
```

`bilinear15_reference.py` is the independent reference implementation copied with its original hash recorded in `manifest.json`. Inputs and native outputs remain in the two original native run directories and are SHA-256 checked before analysis. The separate source-117 replay is used only to confirm exact F-high/F-low FP32 words.

This is a one-point constitutive diagnostic, not a physical step or coupon qualification. It evaluates one declared nonzero deformation direction, not a complete nine-direction smooth-branch finite-difference/noise-floor suite.
