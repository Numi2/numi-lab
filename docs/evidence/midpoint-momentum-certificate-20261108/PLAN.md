# Independent midpoint momentum accounting

Before changing the reference, record the requirement: for the existing n=2,
10 ms, h=15.625 microsecond FP64 midpoint trajectory and unchanged model,
independently form, at every node and accepted step,
`m_i (v1_i-v0_i) - h m_i a_i(midpoint)` from the final states and midpoint
forces. Record the maximum Euclidean norm over nodes and steps. Separately form
the global momentum change against gravity and contact impulse, not the sum
of local norms. Both must be finite and at most 5e-10 N s per step for this
fixture. These bounds are diagnostic and do not replace force or mesh gates.

Add an adversarial arithmetic regression with opposing local residuals: the
signed global balance may be zero but the local maximum must be nonzero. Reject
nonfinite inputs. Preserve the existing 0.25% temporal force and impulse gate,
clearance, energy and work checks. Run existing CPU reference tests after the
change on ssh numi; retain source revision, binary identity and output. No Matter
runtime implementation or production force qualification follows.
