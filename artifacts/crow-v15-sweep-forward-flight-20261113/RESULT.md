# Wing-sweep sensitivity: forward motion with recontact, failed screen

Both push and matched hold completed all 500 first-episode intervals, with accepted state/action/contact parity through tick 345 and valid wing target history. The eight environments are correlated and use a development layout and seed. No waypoint was reached.

With +0.16-rad commanded bilateral sweep, push lanes 4 and 5 moved +0.593 and +0.566 m in world x over ticks 400–500, and averaged +0.302 and +0.288 m/s over that interval. They nevertheless recontacted at ticks 394 and 401 and recorded 3 and 10 toe-speed threshold substeps. Root x velocities at tick 500 were approximately +1.20 and +1.08 m/s, so these results are not bounded-speed flight. Actual sweep reached the modeled +0.20-rad joint limit; mean modeled aerodynamic vertical force over ticks 401–500 was only about 4.06–4.08 N for the 0.45006 kg model. No nonfoot normal impulse was recorded in those lanes, which does not establish safe flight or landing.

The -0.16-rad pair flew backward by about 2.9 m; the oscillatory-sweep pair remained airborne through 500 but moved backward about 0.73 m. No pair passes the predeclared screen. In the forward lanes, sampled aerodynamic work over ticks 476–500 was about -7.82 to -7.95 J and flap effort-displacement proxy was +23.4–23.5 J; these are numerical model quantities, not biological muscle or aerodynamics measurements. Flap efforts stayed within the research 3-unit bound and reached it. The sweep effort was not observed by the flap-effort logger.

The change demonstrates a directional sensitivity of this research model, not a qualified controller, leg-powered takeoff, biological flight, route completion or supported landing. All failures and matching controls are retained.
