# Capped release geometry on the earlier-balance trajectory

At candidate tick 260, native v15 forward kinematics gave foot, front-toe and rear-toe perch gaps of 1.28, 13.47 and 1.97 mm. Local centered derivatives in mm/rad were approximately (-146.3, -150.1, +38.1) for hip, (-83.2, -87.0, +40.1) for knee, and (-23.2, -27.1, +20.0) for ankle. Unperturbed group gaps independently agreed with recorded poses to within 8.6 nm across the sampled states. The foot is actually the tightest clearance at this selected instant, so simply increasing knee or ankle correction can worsen the foot gap; any further target change needs coordinated joint motion.

The original local analysis failed to serialize a NumPy boolean; the corrected script converts it explicitly. Both versions and the failed log are retained. These are static local derivatives, not dynamic contact, a reachable trajectory, or evidence of stable support. The exact native binary/library hashes and input poses are recorded.
