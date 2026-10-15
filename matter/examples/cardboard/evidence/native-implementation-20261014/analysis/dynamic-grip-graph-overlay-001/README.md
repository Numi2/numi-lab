# Dynamic-grip selector-5 overlay (draft 001)

This patch extends the existing bounded graph-Schwarz conditioning-only preconditioner to tolerate per-environment FEM fixed-mask changes. It constructs immutable local graph patches over all potential FEM nodes, then at each linearization inserts identity rows for constrained coordinates and uses the exact free-free principal block for currently free coordinates. The actual production FEM operator, residual, acceptance certificate, and source-authored coarse basis are unchanged. The coarse basis is projected per environment; newly released nodes can lack coarse-mode support, so conditioning can be weaker.

Admission remains limited to the pre-existing immutable, explicit, contact-free FEM scope and graph arena bounds. Candidate tags 0 (free), 1 (authored fixed), and 3 (temporary grip) are accepted only with positive finite mass and consistent inverse mass; tag 2 and other unsupported tags fail closed.

## CPU verification

The layout regression passes 20 checks. It covers potential-node topology coverage, two distinct environment masks, releasing initially source-fixed nodes and acquiring temporary grips, invalid tags and mass/inverse-mass rejection, the restricted free-free/identity matrix contract, and zero correction on constrained rows. The printed `synthetic_allocation_shape_*` values are a synthetic count fixture only, not a claim of exact source-182 patch membership.

The actual source-182 connectivity was separately passed through the production layout helper with all 882 nodes marked as potential. The measured layout is 9 patches, 2,049 patch-node entries, 6,147 scalar DOFs, 4,718,601 matrix floats per environment and a 75,734,380-byte two-environment graph arena. This is layout sizing only.

Runtime and native-checker Objective-C++ syntax checks pass. The integrated Metal build and native acquire/release fixture remain for root verification. No GPU was run for this overlay.

## Application

The patch is based on the immutable snapshot recorded in `manifest.json`. Verify the listed base hashes and apply `patch.diff` only to that snapshot; rebase and rebuild before using another source revision.
