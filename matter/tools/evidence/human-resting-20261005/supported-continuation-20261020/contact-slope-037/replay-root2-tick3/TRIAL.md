# Corrected failure selector 037

GPU nm_complete_microstep increments completedMicrosteps only while status is successful. Therefore failure count3 means ticks0,1,2 succeeded and the failure is tick3. The tick2 replay retained pre-failure contact data and must not diagnose the terminal failure. Reproduce the same three-step request with root2/tick3/slot2624; no physics or binary changes. Require identical accepted CSV and failure tuple and a matching failing trace before interpretation.
