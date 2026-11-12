# First preflight did not run

The first candidate-20261126-rebound-aero attempt exited before simulation because the copied run script looked for rebound policy packs in the new observer build directory. It retained command.json only. The script was corrected to use the existing exact rebound packs and the corrected attempt uses a new output directory; no trajectory or measurement was produced by the first attempt.
