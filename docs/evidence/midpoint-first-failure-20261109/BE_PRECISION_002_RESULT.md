# BE precision 002: complete, replayed, not force qualified

The restricted, default-disabled precision candidate completed 10 ms at h=1 microsecond on the n=2 (27-node) fixed-plane fixture: 10,000 accepted steps, zero rejections. The previous uncorrected BE candidate rejected before contact at step 3,186; the compensated candidate passed that state. Exact restoration and replay, continuation replay, forced rollback and fingerprint mismatch rejection passed. Minimum gap was 87.228 micrometres, maximum independently evaluated local scaled residual 0.0007591, and maximum global momentum imbalance 5.79e-12 N s. The same geometry, physical parameters, area-weighted contact law and residual policy were used.

The complete accuracy analysis against the temporally qualified n=2 FP64 reference fails the registered gates:

| Measurement | Error | Limit | Result |
| --- | ---: | ---: | --- |
| 1 ms-binned force | 5.1849% | 0.25% | fail |
| Sampled peak force | 0.7066% | 1% | pass |
| Total impulse | 1.1447% | 0.25% | fail |
| Cumulative impulse | 1.1447% | 1% | fail |

The reference and native sampled peaks were 0.724674 N and 0.719554 N, respectively. Contact work was -7.41e-8 J and the separately computed impulse-midvelocity work -1.48e-7 J; these quantities are not a complete energy balance. The full trajectory, replay checks, binary/reference hashes, analysis and logs are retained with prefix `be-precision-002-full`. The observed wall time includes diagnostic snapshots and replay and is not a performance comparison.

This result supports the precision diagnosis for the earlier noncontact rejection, not promotion of the new option or evidence of spatial accuracy. The next finer-step test and its gates were registered in `BE_FINE_CONTROL_PLAN.md`. The updated candidate adds narrower material admission and must separately pass its instrument parity and scope checks before that run.
