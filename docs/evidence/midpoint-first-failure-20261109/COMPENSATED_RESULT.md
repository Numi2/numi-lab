# Compensated midpoint diagnostic: incomplete, disabled

This is an isolated research prototype, disabled unless explicitly selected. It
is not a qualified integration method, and does not enable the earlier
augmented-Lagrangian or interval-local CCD experiments. The unchanged law,
physical tolerance and budgets were used. The authoritative velocity was
represented by two FP32 limbs on the GPU, with both limbs included in snapshot,
restore and rollback; unsupported frozen-load and external digest paths fail
closed. The 14 default regression tests passed before the final frozen-load
scope guard was added. Exact transaction checks in the prototype include both
velocity limbs, but broader state-consumer qualification remains outstanding.

At the n=8, 3.90625 microsecond step, the 204-step diagnostic request completed
and forced rejection/replay restored the physical state exactly. A 10 ms request
accepted 204 steps and rejected the next step before contact, with exact
restoration and replay. It therefore does not establish accurate contact response,
spatial accuracy, or a performance benefit.

The selected-step capture took its inputs before rollback. The GPU full-step
momentum norm was 1.98680536e-13 N s against the unchanged 1.37214662e-13 N s
limit; independent FP64 evaluation of the same equations gave 1.98640651e-13 N s.
Twice the stage residual norm was 1.98680421e-13 N s, while the reflected
endpoint correction norm was 4.07e-30 N s. The residual is dominated by the
stage, not endpoint reflection, in this retained failing state. The maximum
kinematic defect was 1.31e-20 m. These observations do not establish why the
stage solve stopped or whether any other trajectory will converge.

The capture executable and metallib were saved before the subsequent rebuild;
their hashes are in compensated-capture-204-inputs.sha256. The first 001 probe
executable was overwritten by a subsequent build. Its recorded hash is retained
in compensated-001-inputs.sha256, but the exact executable was not archived;
that provenance limit applies to the 204-step and full-rollout 001 records. The
selected-step capture and its independent residual are separately reproducible
from their retained executable and source. Failed rollout and capture data are
retained unchanged. These results are not timed performance measurements.

A separate trace of the same saved-binary run reproduced the identical
accepted-state hash and final capture. Its reported nonlinear stage norm fell
from 2.7685e-6 to 3.5341e-13 and then 9.9750e-14 N s, but remained at
9.9340e-14 N s from iterations 4 through 15; the stage limit was approximately
6.8607e-14 N s. The recorded line-search scalar was 1 throughout, and the
solver reported a much smaller final linear residual metric. This suggests a
stage-update precision or consistency stall; the trace alone does not prove a
cause, and its linear metric is not an independently recomputed b-Ax. A
separate independent tangent-vector and correction audit remains necessary.

The read-only rich trace retained the stage and correction arenas at iterations
3 and 4. Independent host-side FP32 addition shows that all 729 proposed
vertical corrections in each of these iterations round back to their existing
stage velocity. All 729 corrections on each horizontal axis change the stored
value. This is direct evidence of a vertical update representation limit at the
captured stage, not proof that a particular alternative will meet the contact
accuracy gates. The analyzer and raw snapshots are retained. Its evidence is a
rounding check, not an independently recomputed linear residual or tangent.
The default regression suite was repeated after adding the frozen-load guard and
passed 14/14; the subsequent trace-only change is separately opt-in.
