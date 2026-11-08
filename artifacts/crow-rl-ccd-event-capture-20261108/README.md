# Selected CCD event captured and replayed

The optional protected capture now handles zero-time replay and advance-budget
failures, keeping the selected pair and exact projected endpoints before the
first failing event. Capture cannot mutate admission or physics. Both retained
clamp32 and release64 trajectories, typed failure and rollback/world states are
byte exact; the successful open control is exact and generates no capture.
The8192-pair grid and old unresolved-query capture replay exactly, and the
native contact owner regression passes.

Both new selected-event captures replay their stored first native candidate
exactly. Clamp predicts contact at146.369microseconds while returning lower0;
release predicts307.597microseconds while returning lower0. Their cursors
repeat zero selected time3times and fail. The independently calibrated FP64
fixed-box reference confirms first contact at fractions0.234181 and0.984367
of the remaining interval; maximum257-query distance disagreements are
22.84nanometres and24.80nanometres. These are measured projected geometries,
not a complete mechanical capability.

The next model revision will test using the certified supporting-plane contact
time for the translating primitive. Tolerances, event budgets, rotating-body
path and all other physical authority remain subject to unchanged controls.
Full Release build passes. The first preparation invocation had an untracked
script that failed its clean-tree guard before any trial; it and its log remain
retained. No physical condition or observation was retried.
