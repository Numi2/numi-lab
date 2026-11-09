# Diagnose native repair failure at step 407

The first native repair accepted the retained step 203 and independently
passed its original physical residual limit; the first 203 prior trajectory
steps matched. The subsequent 10 ms request nevertheless rejected at step 407,
still before contact, with nonlinear status 10. Replay, forced rollback and
restored physical state passed. Preserve the full failed rollout.

Capture the new failure at the existing certificate boundary, using the same
binary, shader, fixture, tolerance and step. Confirm exact restoration and
independent FP64 agreement with the reported residual before interpreting the
capture. Then run the already described independent reflected-stage rounding
iteration on the retained state to distinguish an inadequate correction from
its representability limits. Retain every output; no certificate relaxation
or contact-law changes are allowed. This diagnostic does not establish contact
accuracy or qualify a full rollout.
