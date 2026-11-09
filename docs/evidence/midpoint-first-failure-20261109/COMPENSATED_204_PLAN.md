# Compensated midpoint first-failure diagnosis

The first compensated-velocity native 204-step request completed and restored
both velocity limbs on forced rejection and replay. Its next 10 ms request
accepted 204 steps, then rejected step 204 before contact with a full-step
momentum norm above the unchanged physical limit. That failed rollout is not
force qualification. The opt-in implementation remains isolated from the
production branch while its transaction and state consumers are audited.

Extend the existing selected-step diagnostic format to retain both accepted
and candidate endpoint low velocity limbs. Record the corrected midpoint
stage, independently evaluate the same full-step equations in FP64 with both
limbs, and compare to the device certificate and exact restored physical
state. An absent low limb, mismatched fingerprint, capture after rollback, or
inconsistent residual invalidates the diagnostic. Do not alter the material,
contact law, physical limit, stage tolerance or iteration budgets. Distinguish
stage-residual nonconvergence from reflected endpoint representation and
retain the capture and failed rollout regardless of outcome.
