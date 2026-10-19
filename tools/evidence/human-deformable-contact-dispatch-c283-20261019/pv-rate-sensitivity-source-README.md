# Aortic cavity wall PV rate sensitivity

This is a paired four-step, 0.25 ms wall-only circulation run at cycle fraction 0.25. The rate-8 case is a 32-step exploratory run intentionally stopped after four accepted steps; the rate-0 case completed four steps. Both share the same app, library, MSL, payloads, and wall setup. Exact hashes and row-level values are in `comparison.json`.

At the initial sample, geometric cavity volume is 18.4182826 mL. It decreases in both runs over the observed startup interval. Rate-0 differs materially from the runtime-default rate-8 case: cavity pressure differs by -616.6 Pa at step 1 and by -115.1 Pa at step 4; wall displacement and geometric volume also diverge. This supports treating rate 0 as a sensitivity control, not as a production setting. The short interval does not establish systolic response or full-cycle physiology.

The trace reports live network pressure, the cavity hydraulic volume row, FEM geometric volume, and wall displacement for the same accepted steps. This is coupled-state evidence, not physiological validation.
