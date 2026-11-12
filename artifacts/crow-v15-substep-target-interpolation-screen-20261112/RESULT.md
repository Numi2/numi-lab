# Target interpolation attempt: inactive path

The sham reproduced all 26,400 observed joint, contact and target samples exactly. The candidate also reproduced the same states and contacts and produced identical boundary second-difference values; it did not achieve the requested 20% reduction. Source inspection shows this candidate modified bounded/coupled drive kernels, while the active experiment uses the implicit drive refresh kernel. The intervention therefore did not test whether interpolation helps. No physical gait improvement can be inferred. The complete matched runs and the negative result are retained.

The extra-history buffer and initialization are an experimental chunk-size-one mechanism, not a production-ready solution. A separate experiment must address the active implicit drive path, retain the same sham and verify its effect.
