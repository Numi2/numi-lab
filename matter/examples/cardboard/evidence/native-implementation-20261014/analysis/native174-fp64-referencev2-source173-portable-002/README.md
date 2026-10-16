# Portable compensated FP64 reference package 002

This is a new version following seal 001; seal 001 is unchanged. The correction fixes a retained helper bug in the legacy compensated FB row: the quotient branch computes a compact masked vector, so the destination assignment must use that vector directly rather than applying the full-length mask a second time. The new source173-local test exercises both mixed direct/quotient branches through the helper, residual, Jacobian, and projection paths, plus the v2 projection/tangent path. The fixture is a small sibling Python module; the test discovers the existing equation reference by sibling filename and has no absolute workspace path.

The five-file additive patch was dry-run and applied to a temporary source173 tools tree. All five outputs were byte-identical to this package. The permanent sibling test passed 5/5 using qualified Python 3.10.12, and py_compile passed. The source equation emitted two NumPy RuntimeWarnings from its eager np.where evaluation of a zero-denominator branch during projection; outputs remained finite and tests passed. This package does not change or suppress those source warnings.

The run018 result remains bound to the exact portable module snapshots it used and retains 16/15,552 row-directions without adjacent passes on the original epsilon grid. The separate run019 focused quarter-grid follow-up found adjacent passes for 15/16 of those selected cases; F[2,0], paper row/tet 172 remains unresolved with only isolated passing grid values. Neither focused result establishes a measured absolute noise floor or changes run018's original score.

This is CPU reference software evidence. It is not native Metal, calibration, coupon convergence, or physical qualification.
