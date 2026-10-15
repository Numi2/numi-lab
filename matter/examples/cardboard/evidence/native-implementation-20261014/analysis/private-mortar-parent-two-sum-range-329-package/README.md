# TwoSum operand-range supplement

This is a separate supplement to the sealed 327 common-translation proof. Metal may flush subnormal arithmetic, so an error-free TwoSum expansion is not a valid proof for arbitrary tiny/huge endpoints. The helper now accepts only finite endpoint operands in the closed range [1e-20, 1e20] (zero is valid) before evaluating endpoint differences; out-of-range motion falls back to CCD. No contact tolerance or threshold was changed.

The CPU suite passes 32 checks, including subnormal, below/above-range, zero, and a normal-range 2^-30 rounding adversary. Full Metal source compile/link passed in the prepared source. No GPU/native run was made.

Apply after the 327 patch with `git apply two-sum-range.patch`. See `manifest.json` for exact source and file hashes.
