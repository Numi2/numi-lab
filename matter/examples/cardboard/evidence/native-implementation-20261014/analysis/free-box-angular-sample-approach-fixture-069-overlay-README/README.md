# Angular contact sample fixture overlay (source 069)

This checker-only overlay repairs two synthetic fixture setup defects identified by the raw source-069 native sample capture. It does not change production runtime or Metal, contact thresholds, or acceptance gates.

## Diagnosis

Native case 217 bound to the raw-diagnostics checker and source 069. Each contact sample and the inertia sample remained all-zero with success status, while node and rigid state readbacks were populated. The earliest fixture defect is that zero-initialization left `proxy.adaptiveObjectIndex = 0`. The production evaluator therefore treated FEM object 0 as the rigid proxy's adaptive owner, saw its active representation was FEM rather than RIGID, and returned without writing a sample. The checker now explicitly sets `adaptiveObjectIndex = NM_INVALID_INDEX`, matching a non-adaptive free body.

The fixture also set the stationary node velocity to zero but authored the box point velocity as `-0.5*n`, which is separating for the evaluator's outward box normal. The corrected body velocity is `+0.5*n - omega×r`; a host-side assertion checks the intended approaching relative velocity before production evaluation. This second correction is necessary for the positive angular-barrier sample the test requires once the adaptive-owner early return is removed.

Raw status/sample diagnostics and fail-closed derivative gates remain. Invalid samples still fail the checker; independent cases continue only when their prerequisites are valid. The native output from case 217 is preserved and hashed in `source-overlay.json`.

## Validation

CPU Objective-C++ syntax check passes against frozen source-069 headers with `-Wall -Wextra -Wpedantic -Werror`. No Metal build or GPU execution was performed. A fresh native run is required to verify sample admission, direct residuals, contact tangent, and inertia action.
