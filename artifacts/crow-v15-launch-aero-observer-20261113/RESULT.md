# Launch-window aerodynamic observation

The pre-registered replay completed successfully. For all eight lanes, state, velocity, action and contacts matched the prior replay exactly through interval 500; all recorded drive-target, contact, leg and toe samples in intervals 335–500 also matched. Physics input hashes matched, no observer errors occurred, and every expected aerodynamic substep in intervals 335–400 was captured.

For the two leg-push lanes, modeled net vertical aerodynamic impulse during intervals 371–390 was 1.1971 and 1.1890 N s respectively; the corresponding modeled leg-push increments versus held-leg controls were 0.09769 and 0.03782 N s of foot normal impulse. The earlier held-leg controls also became airborne, so these observations do not establish a necessary leg-powered takeoff. The aerodynamic force model has not been biologically calibrated. This is an instrumentation and attribution result, not a flight or mission qualification.

The raw observer output and replay are retained here. `observed-window.json.gz` is a derived extraction reproducible with `extract.py`; `analysis.py` also reads the reproducible push-window extraction documented in the leg-wing attribution evidence.
