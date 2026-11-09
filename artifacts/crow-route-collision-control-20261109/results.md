# Collision-enabled route control

The collision-enabled matched route trial failed its screen: all 64 first episodes physically terminated, with zero failed simulator steps. Forty-eight had already triggered all five waypoint gates, 13 reached two, and three reached one. The final-published-substep contact records implicate the crow body and perch in 48 lanes, the right wing and second slalom in 13, and the right wing and gate in three. Some perch records also contain foot contacts. These are candidate contacts from the final published substep, not proof of the first fatal microstep; see contact-summary.json and individual pre-contact q/v.

This overturns any interpretation of the collision-free waypoint result as successful navigation. In particular, crossing the final gate with body/perch impact is not supported landing. The controller and actor tensors were unchanged; explicit contract rebinding accounts for the collision-enabled world.
