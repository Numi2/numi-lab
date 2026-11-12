# Initial observer launch rejected

The first attempted observer launch used the gait host's model object rather than the open-flight scene object. The existing policy fingerprint check rejected the run before any rollout; it is retained as `candidate-20261126-rebound-flexed`. The host was then relinked with the same open-flight model object as the reference. The corrected observer run must independently demonstrate full trajectory/contact parity.
