# 547 critical23 endpoint input audit

The audit compares each serialized row to the immutable center row from native541-center-input-001/input.json, not merely to its accompanying reference file. It reconstructs F from exact FP32 high/low words, checks center F plus or minus 2^-p H, validates accepted-history words, and reports FP32 rate-gradient rounding separately.

The superseded 547 v1 and earlier 541 critical23 batch are retained as invalid finite-difference endpoint inputs due to aliased NumPy FP32 views. The separate 541 center-only native replay remains usable. Corrected 547 v2 passed CPU admission; this is input/admission evidence, not native material qualification.

Reproduce on macmini with:
 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 audit.py

The script expects the bound external inputs described in report.json.

The report and script are bound by manifest.json. External inputs remain in their original evidence folders; their exact hashes and sizes are listed in external-bindings.json. The 547 v2 CPU-admission receipt reports 1472 rows and native_gpu_executed=false.
