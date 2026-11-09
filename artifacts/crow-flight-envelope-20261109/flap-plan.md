# Exploratory differential flapping authority probe

With roll feedback preserving lateral stability but body yaw saturated, test whether the existing differential flap action can supply additional yaw without excessive sideslip. Compare +0.20/-0.20 and -0.20/+0.20 normalized left/right flap-action offsets against the roll-1.5 control, from 6 s onward and clamped to [-1,1] before normal actuator response. Preserve v15 body and all aerodynamic, actuation and solver limits, the fixed speed trim, roll feedback, commanded rates, matched 32 starts, seed 20261111 and horizon.

Prediction: at least one sign reduces -0.03-lane median maximum heading error by >=0.10 rad relative to roll-1.5 while no physical terminations occur and median maximum lateral speed <=0.20 m/s. Also report speed and height errors and every strict-screen result; zero failed simulator steps and complete first-episode samples are required. Stop after these two arms. No unused layouts or final-test seeds are used.
