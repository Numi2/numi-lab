# Hip-target direction kinematics audit

At the default pose with the root fixed, increasing both hip coordinates by 0.001 rad moves the left-foot center approximately 0.134 mm backward and the whole-bird COM 0.0077 mm backward. If the foot were anchored, the hip motion would tend to shift the root forward, consistent with the sign of the existing correction for an aft root. The local geometry does not establish anchoring or explain the observed fall. It suggests testing bounded response strength, rather than assuming the control sign is reversed.
