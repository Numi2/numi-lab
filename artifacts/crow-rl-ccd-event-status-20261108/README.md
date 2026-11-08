# First CCD event failure preservation

The CCD initializer cleared event counters after an environment had already
failed, while later statically encoded substeps skipped its physical work.
It now returns immediately for non-success status. The accepted dynamics,
rollback and public failure stamp are unchanged on the retained clamp failure.

The production native probe checks both hybrid and disabled CCD, all thirteen
non-success codes, two generations and an out-of-range lane: 56 complete-buffer
preservation checks pass. All four healthy initializer records match the old
shader exactly; a repeat is exact. A complete successful open-jaw control has
identical traces, final q/v/scene and scene trajectory. The native contact
regression passes. The initial build invocation lacked CMake on PATH; its log
is retained, followed by the successful full Release build and execution.

With first counters preserved, clamp lane 0 fails at tick 32/substep 19 after
three zero-time event replays, with no time consumed. Lane 2 fails at substep14
after four event passes, with 0.000514183 seconds remaining. The earlier zero
counters did not demonstrate a contact-solver failure. These are event-budget
failures; their responsible projected candidate still requires protected
capture before changing event timing. No event budget or tolerance was raised.

This is diagnostic correctness, not completed physical beak qualification.
The learned policy and protected deployment are unchanged.
