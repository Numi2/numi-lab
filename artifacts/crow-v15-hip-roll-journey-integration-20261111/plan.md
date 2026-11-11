# Experimental hip-roll body journey integration

Construct an isolated journey host that swaps the two additional experimental hip roll joints into the existing two-segment-toe body. Preserve the frozen flight controller and use the existing wide-body serial ABA backend. First perform contract checks only; changing joint and body indices can affect task bindings, sensors, contacts, policy fingerprints and rendering. No autonomous, flight or landing qualification follows from build success. Keep the original host and controller untouched.
