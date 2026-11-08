# Preallocated control correction

The added test initially compared every row of the [0, 1] and [0, 0] policy-mask runs for equality. This was wrong for environment one in the preallocated-two-environment arm: it already performed a step, so with no reset its correct next phase/action are 0.04/0.010, not 0.02/0.005. The failing CTest and both unmodified native records are retained. The genuine growth arm first creates environment one during its second call and is the relevant zero-initialization comparison.

Correct the assertion to test the continuing first row in every arm, the new second row under growth in both masks, and the preallocated second row according to whether it resets. Do not change the runtime implementation or numerical tolerances in response to this test error.
