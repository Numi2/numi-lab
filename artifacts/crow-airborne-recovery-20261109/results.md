# Airborne-start recovery result

72 of 80 lanes passed the preregistered 6–14 s recovery screen, with zero physical failures and zero failed simulator steps. All eight +0.10 rad pitch lanes failed because the sampled forward-speed error at 6 s was 0.2266–0.2396 m/s, above the fixed 0.20 m/s limit. The other nine perturbation conditions passed 8/8 each. A representative failed lane had forward velocity 1.427, 1.253, 0.936 and 0.727 m/s at 2, 4, 6 and 8 s; thus it eventually approached the target, but the prespecified recovery deadline was missed. The selected lane-one step trace, source states, source identities, all first-episode outcomes and compressed samples are retained.

The start reconstruction uses measured q/v and filtered actions but does not restore every hidden solver or task-history variable; it is a meaningful reset perturbation screen, not proof of continuous flight or deployment qualification. The controller is not qualified or frozen.
