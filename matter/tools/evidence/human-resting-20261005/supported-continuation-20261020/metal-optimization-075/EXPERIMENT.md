# Experiment 075: distributed solver and active contact work

Registered before new native measurements. Exploratory implementation study;
no physiological, timestep or reduced-model qualification implied.

Control: ea2cbd84 native binary and preserved compatible Matter metallib.
Treatment: explicit work domains, telemetry, CGS2 tiled reductions and active
radix incidence; individual ablations retain exact same scene inputs/root dt.
Question: does distributed work lower actual accepted-root cost while preserving
true residual, transaction and existing 1% endpoint metric gates?
Prediction: combined treatment improves median 4-root body throughput by at
least 2x; refuted below 2x, invalid if a numerical gate fails. No claim of 0.1x
until a supported fully coupled sustained run establishes it.
Units: three separate matched native process pairs, identical initial scene,
0.125ms root, four roots per process; order C/T, T/C, C/T. Initialization and
process overhead reported separately. First one-root smoke is software-only,
then short mechanics ablations; do not count roots as independent trials.
Counters: sum used columns across all cycles vs separate maximum status;
whole-Newton stage timestamps plus kernel samples every 127 with phase rotation.
Calibrate stage/sampling overhead using matched instrumented/uninstrumented
runs. Keep failed/censored trials. Stop a native process after 180s without an
accepted root; investigate actual failure before repeating. No concurrent GPU
work. Required checks: finite state, accepted roots, tissue audit, no assistance,
true residual and rollback regressions. Existing 1% comparison remains fixed.
Large-timestep/reduced-model and supported-rest qualification are separate
follow-ups, not allowed to inherit these comparisons.
