# Explicit bounded flight-command boundary: parity screen passed

With the same scripted command provider, used seed 20261126, training layout, teacher off and no resets, all 14,408 recorded state records across eight lanes match the verified prior combined-turn flight in q, v, policy action and contacts, including the initial samples and complete 1800-interval episode. All registered continuous-contact, actuator, toe, heading, turn, altitude and substep gates also passed. The copied raw trace, replay and TSV match the native-host hashes.

This establishes regression parity for the refactor's existing schedule only. Speed is fixed at 0.35 m/s and the height and turn commands remain scheduled; no arbitrary command envelope, airborne disturbed-start recovery, general navigation interface, flight freeze, leg-powered takeoff, toe grip, landing or held-out mission is qualified by this result.
