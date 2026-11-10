# Flight time-step refinement results

The original body passed 8/8 lanes at both 8 and 16 substeps; the articulated-toe body passed 0/8 at each, with no solver or physical failures. In straight flight the maximum forward-speed error was 0.079 and 0.076 m/s for the original, versus 0.338 and 0.539 m/s for the toe body at 8 and 16 substeps respectively. The predeclared gate is 0.20 m/s. This is a numerical sensitivity and a tracking failure, not proof of the accurate solution or an isolated cause. No version is promoted; walking, takeoff, landing and autonomy remain unqualified.

The original 4-substep two-seed results are retained in crow-v15-original-free-flight-20261110 and crow-v15-toe-free-flight-20261110. Runtime dependencies used here are archived in those experiments; the newly rebound zero policies are included below. Each records the actual world fingerprint and contains only zero output weights.
