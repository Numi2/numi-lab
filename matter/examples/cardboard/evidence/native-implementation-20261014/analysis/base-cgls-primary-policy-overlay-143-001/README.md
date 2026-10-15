# Base CGLS primary-policy overlay 143

This package is a narrow, default-off candidate against the frozen source142 snapshot. It enables the existing truncated CGLS direction as the primary direction from an explicitly selected Newton index, only on the selected all-Newton FEM control step. It does not change the physical operator, material law, residual tolerances, Newton certificate, Armijo rule, accepted history, or rollback path.

The device selector runs after base residual/status capture. It marks an environment only when status is healthy, the original Newton linear certificate is present, the base residual is strictly above the existing no-op bound, the inclusive start index is reached, and no selector marker is already set. CGLS then generates the existing direction before a single ordinary Armijo pass; `fallbackAllowed` is zero. Environments already within the no-op bound are not selected, retaining the unchanged certified-no-op path. The original FGMRES certificate is cleared by the existing CGLS installer because it certifies only the original Newton direction.

The runtime setting is `NM_FEM_BASE_CGLS_PRIMARY_AFTER=<canonical decimal Newton index 0..256>`. It is rejected unless CGLS iterations are configured and `NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=<control>:all` selects the already-bounded path. With the setting absent, the old exact-index/all-Newton fallback behavior and trace payload remain unchanged. The enabled log states the selection predicate as `healthy_base_above_noop_bound_with_original_linear_certificate`.

## Files and application

`overlay.patch` contains only the two changed production sources: `matter/src/runtime.mm` and `matter/src/metal/base_cgls_runtime.metalinc`. The test is package-only and is not part of the production patch. Apply from the repository root that matches the bound source snapshot. `manifest.json` binds the baseline source, changed files, patch, exact selector body executed by the CPU check, and test result.

## Verification

Command: `python3 tests/test_base_cgls_primary_policy_contract.py` from `overlay/`. The check compiles and executes the selector body extracted verbatim from the changed Metal include, adapting only Metal qualifiers/attributes and providing ABI stubs. Cases cover start boundary, healthy/certificate requirements, residual/no-op boundary, nonfinite values, object/environment scope, and preservation of a pre-existing marker. It separately audits host parser constraints and ordering against source text; it does not claim runtime coverage of the host environment parser.

No project build, Metal compilation, GPU/native execution, or end-to-end solver qualification was performed. Root should integrate the patch on the selected source and build/test it before any native measurement.
