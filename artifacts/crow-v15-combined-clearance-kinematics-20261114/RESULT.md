# Combined trial clearance geometry

At candidate zero-based tick 265, native forward kinematics gave foot/front/rear gaps of 0.948, 12.981 and 4.626 mm. Centered local derivatives in mm/rad for hip, knee and ankle were approximately (-146.7, -151.0, +33.3), (-83.9, -88.2, +39.5) and (-23.6, -27.9, +20.4), respectively. Recorded-pose gap crosschecks agreed within 9 nm across sampled lanes and ticks. The native executable and library hashes match the earlier audit.

The released foot is the limiting geometry at this sampled state. A hypothetical additional -0.05 rad hip displacement gives a first-order foot-gap increase of about 7.3 mm and rear-gap decrease of about 1.7 mm; the sampled rear gap is 4.6 mm. This is a static local estimate, not a dynamically reachable trajectory or a guarantee of stable support. The next paired physical experiment must measure actual motion, contacts, limits and balance.
