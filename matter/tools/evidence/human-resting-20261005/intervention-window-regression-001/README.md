# Complete-breath intervention measurements

The earlier 60 s study retained a five-second recovery window whose ventilation
comparison depended strongly on the partial breath at the window boundaries.
Its PaCO2 result and failed-to-establish-recovery conclusion remain unchanged.

The existing study adapter now supports `--window-s` (default 5 for compatibility)
and reports additional complete-breath ventilation/rate metrics. These use the
native accepted breath counter and interpolate the corresponding zero crossings
of accepted airflow. No requested controller frequency or prescribed curve is
used. A window with fewer than two inspiratory boundaries explicitly reports
unavailable; it does not report zero ventilation. Skipped/inconsistent accepted
breath transitions fail admission.

Four arithmetic/admission tests passed on the SSH Apple M4 Pro Mini: partial-edge
invariance, unavailable short windows, rejected skipped transitions, and positive
flow integration across a zero crossing. Tests are not physiology validation.

The retained-trace diagnostic is explicitly post hoc. None of the old five-second
recovery windows contains two inspiratory boundaries. Over the wider [35,60] s
interval, complete-breath ventilation is 5.728 L/min in control and 6.918 L/min
in treatment (three and four complete breaths, respectively). Those unequal
short-run values do not establish recovery. The next paired whole-body run must
declare longer windows before execution and retain both fixed-window gas
measurements and complete-breath ventilation comparisons.
