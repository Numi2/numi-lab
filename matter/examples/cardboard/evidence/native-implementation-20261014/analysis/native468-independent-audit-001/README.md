# Independent audit of native468 analysis 002

This package independently rehashes and checks the complete terminal attempt and the source analysis package. It does not run or modify the native solver.

The run completed normally with exit code 2 after 20 bent loading controls were accepted and control20 (zero-based) was rejected with status10. The held-reference environment accepted its 21 attempted controls independently. There was no timeout, process-group termination, or capacity abort. The planned 60-control loading/hold/withdrawal/release/relaxation sequence did not complete.

The audit verifies all 894 prediction-bound input aliases, all 42 observation rows, all 77 non-input terminal files, and all 23 rejection-conditioning arrays. It independently parses stderr and checks 64 Newton final-cycle summaries for the rejected control, 64 CGLS per-Newton summary records, and the primary CGLS policy window at Newton48 through63. It also checks every structured trace event parses and the summarized events match the original stderr.

The rejected bent CSV row repeats the prior accepted row's geometry/material observables (including min/max J, displacement, speed, kinetic energy, material-state change, measured angle, and fixed-node count), while retaining a distinct rejected-candidate residual and correction certificate. The material-state sidecar labels the state as rolled back and has raw acceptance flag0. These are different state roles; the rejected row is not one accepted physical state.

The report remains an integrity audit of a failed incomplete attempt. It does not qualify solver correctness, numerical convergence, a complete loading cycle, physical assembly, or crush/impact behavior. The initial effective ambient solver environment was not fully sealed by this legacy run; a later live observation does not repair that provenance gap.

Run `python3 audit.py` from this directory to repeat the read-only checks against the retained source run and analysis package.
