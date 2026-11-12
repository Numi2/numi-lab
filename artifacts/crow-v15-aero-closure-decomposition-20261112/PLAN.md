# Counterfactual decomposition of wing closure

At the identical accepted q/v and on the same GPU timeline, evaluate a separate closure-only copy of the v15 wing kernel into a zeroed diagnostic buffer, after the ordinary wing forces have been computed. The scratch loads do not enter the physics. Compare closure-only wing force and torso-referenced moment against the ordinary recorded wrench, exact state and contact parity, and note that subtraction may also contain previous actuator contributions. This is model attribution, not a direct crow-force measurement.
