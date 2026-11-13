# Smaller sweep trim: failed flight screen

All first episodes completed 500 intervals, the matched state/action/contact prefixes through tick 345 agreed, and target-history and corrected wing-drive checks passed. Tests use one development seed and layout with correlated lanes.

No candidate pair passes the preregistered forward-flight screen: all six push lanes recontacted after release. At +0.04 rad commanded sweep the two push lanes moved -0.65 and -0.63 m in course x over ticks 400–500; at +0.08 rad they moved -0.20 and -0.23 m; at +0.12 rad they moved +0.111 and +0.112 m, below the +0.15 m criterion. First recontact occurred at ticks 394–433. The +0.12 lanes did not reach the +0.20 rad modeled sweep stop; one had three toe-speed threshold events. Both had mean modeled late vertical aero force about 4.17 N, below the roughly 4.42 N model weight. No logged nonfoot normal impulse occurred in the six push lanes.

These results indicate a directional tradeoff in the research model, not a qualified maneuver, biological validation, leg-powered takeoff or route/landing result. The included force-work values and actuator effort-displacement proxy are numerical model quantities, not measurements of a real bird.
