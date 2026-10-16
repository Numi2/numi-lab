# Bounded all-control CGLS selection

This source149-relative overlay adds exact `NM_FEM_TRUE_RESIDUAL_ROOT=all` and
`NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=all:all` selectors. The latter requires the
existing bounded CGLS iteration option and primary policy. Every selected control
resolves to its own control index and repeats the original pure-FEM capability,
positive-mass, Armijo, cached-tangent and true-residual certificate checks. The
physical operator, material update, thresholds, direction installer and rollback
are unchanged. Numeric selectors preserve their prior behavior.

The purpose is to permit a complete coupon attempt once its individual nonlinear
steps can solve. It does not establish that those steps converge.

`tools/prepare.py` constructs the single-runtime-file patch with exact-match
anchors. `tools/test_extracted_selectors.py` compiles the actual extracted C++
selector code: all 44 admission/rejection and numeric-legacy checks pass. These
are parser checks, not solver tests. `tools/assemble_build.py` creates immutable
source153, verifies 787 source and 1,034 configure-support hashes, checks that
only runtime and frozen metadata differ, and builds the probe with four CPU jobs.
The source-bound build passes; it does not launch a GPU simulation. The independent
review binds the same patch, source and build hashes.

All-control production commands must omit singleton CGLS/state/regional capture
options. Those observers select by Newton index and write an exclusive single
pathname; subsequent controls would collide. Use numeric control selectors for a
targeted capture. Full accepted driver histories remain the evidence for complete
coupon attempts.

The driver inherited here still admits at most 64 nonlinear iterations; changing
that operational budget is a separate overlay. Source153 has no native parity or
all-control coupon result yet. The block3 conditioning experiment remains bounded
to contact-free, cached pure FEM with E≤2, N≤4096 and T≤32768. It cannot be applied
to the shipping mesh or nonlinear contact by changing a selector.
