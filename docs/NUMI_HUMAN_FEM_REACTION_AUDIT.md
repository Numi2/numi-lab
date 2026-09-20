# Numi Human FEM anchor-reaction audit

`NumiHumanTendonFEMLoadAdapter` now audits the prescribed-node reaction field
inside the borrowed Human command buffer immediately before the same field is
projected through the articulated body Jacobian. This closes an evidence gap:
reading Matter's private reaction scratch after replay or rollback can observe
restored scratch rather than the reaction consumed by the accepted pass.

The adapter exposes last-pass reaction L1/resultant values and a bounded
per-step history. A post-validation commit kernel writes the history only when
the enclosing Human step succeeds, so deliberately rejected attempts cannot
publish reaction evidence. If an accepted replay reuses a logical step index,
the history retains the stronger accepted reaction instead of allowing reused
runtime scratch to erase it. Diagnostics report the audited step count, minimum
and maximum trajectory L1, and maximum trajectory resultant.

The native `matter.numi_human.tendon_fem_transaction` test requires a nonzero
two-step reaction history in addition to its existing bitwise replay, rollback,
force replacement, contact, and articular-wrench checks. The thoracoabdominal
Human qualification uses the same audit to require a nonzero reaction for all
four qualified 10 microsecond steps.

This audit certifies the exact reaction consumed by runtime coupling. It does
not independently validate anatomical attachment placement, material
calibration, or physiological load sharing.
