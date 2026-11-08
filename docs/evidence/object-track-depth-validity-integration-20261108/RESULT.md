# Integrated object-track depth-validity verification

Integration merge `79d8afd87b62dbc7b210c5a16c3a917b2e6fc598` includes main `4a29404dc` and the prior integrated branch. On Apple M6 the shared Metal shader, renderer, native full-dropout fixture, standalone object-track kernel probe, policy-migration tests and task-program check were rebuilt. All seven targeted CTests passed. The native fixture retains selected geometry while depth dropout is forced, and validates no spurious track in two environments; the kernel fixture separately covers mixed validity and a valid/dropout/reacquired sequence.

This is a scoped observation-admission check, not calibration, learning or physical-step rollback qualification. Complete baseline and candidate evidence is retained in the main report. No performance claim is made.
