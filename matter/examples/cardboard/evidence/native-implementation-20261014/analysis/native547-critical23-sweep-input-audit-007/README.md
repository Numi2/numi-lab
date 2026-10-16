# 547 critical23 endpoint input audit

The audit compares every serialized endpoint to the immutable center row in native541-center-input-001/input.json. It does not treat the accompanying reference as source truth. It reconstructs F from FP32 high/low words, tests the intended center F plus or minus 2^-p H, checks accepted-history words, and reports rate-gradient rounding separately.

Superseded 547 v1 and prior 541 critical23 endpoint batches are invalid finite-difference inputs due to NumPy FP32 buffer aliasing. The separate 541 center-only native replay remains usable. Corrected 547 v2 passed CPU admission; this audit does not qualify native material derivatives.

Reproduce on macmini with the source-bound Python path recorded in report.json. The script expects immutable external source rows and all three input/reference packages at the paths recorded in the script.
