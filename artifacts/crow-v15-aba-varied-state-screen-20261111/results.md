# Varied-state fast-math screen failed

For the same 96 deterministic yaw, velocity and effort variations, both isolated fast-math bodies failed the original 5e-5 scaled GPU-versus-CPU acceleration gate. Maximum error was 6.92255e-5 for the 33-body two-segment prototype and 6.21003e-5 for the 25-body single-segment control. Both maxima were in neck pitch (DoF 20). Repeated GPU outputs matched exactly and all native lanes completed; this is a numerical-accuracy failure, not a crash. These free-space tests cannot qualify physical contact, flight or landing.
