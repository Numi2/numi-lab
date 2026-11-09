# Retained n=8 native midpoint trace integrity

The previously published 10 ms, 1,280-step native midpoint rollout at h=7.8125 µs was independently rechecked for finite fields, complete accepted steps, replay flags, geometric clearance, contact impulse accounting, and exact correspondence with the published trace and analysis. The force and impulse analysis reproduced exactly against the mesh-specific time-refined reference, including its area weights.

The reference-relative maximum error in 1 ms-binned force was 0.07169%; the separately sampled peak-force error was 0.14450%. These measures are not interchangeable. The latter compares the native step-sampled peak to the finer reference peak. This recheck adds no new rollout or performance measurement, and n=8 remains spatially unqualified. The retained execution is tied to its existing recorded binary and source inputs rather than to later changes in this checkout.
