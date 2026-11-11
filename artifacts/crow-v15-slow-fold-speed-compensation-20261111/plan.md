# Bounded speed compensation for slow leg fold

Increase only the high-level speed command by up to 0.05 m/s as the physical leg fold ramps in, capped at the actor command bound of 0.40 m/s. Keep the frozen low-level flight law, toe joints and collisions unchanged. Compare to the 0.35 m/s intended-speed reference, reporting both intended and applied high-level commands. Qualify the whole speed, turn, altitude, self-contact and perturbed-start envelope before integrating with navigation. This is an untested candidate.
