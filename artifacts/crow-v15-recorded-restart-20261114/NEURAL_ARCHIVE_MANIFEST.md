# Neural balance training archive

`crow-neural-balance-v1-20261115.tar.gz` SHA-256: `e3798719d091901e60961dd5f9304eef5ecd734d2a385b43ab3c69f4a28ceb8c`. It contains the initial and updated policy packs, all four retained PPO rollout files and optimizer states, and the final deployment policy. It is evidence of the development training, not of successful balancing.

The balance shader applies all four learned residual actions throughout the forced terminal phase, including its approach; the earlier plan's narrower description of application during contact was superseded to preserve behavior-policy likelihood consistency. The scripted terminal carrier remains active. Final instrumented masked PPO Python SHA-256: `ef3b0f030508abded2c4a05c3a06348ea1cf28422fe2000bf37bd71cc855cd83`; the earlier build manifest predates this instrumentation.
