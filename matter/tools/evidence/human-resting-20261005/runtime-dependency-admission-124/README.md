# Bind implicitly loaded native dependencies

The native study adapter now requires the same seven runtime libraries and
Metal libraries already bound by `numilab_human.resting_run`. Checking only
absolute paths in argv misses libraries loaded by the executable itself.
The existing pre/post-run hash checks then protect these additional bindings.

All 21 adapter tests passed on the SSH Mac mini. The added regression removes
each dependency in turn and confirms rejection before execution. The actual
run123 invocation is admitted; the manual study100 reference is rejected for
its omitted `libmetalrobo.dylib` binding. No simulation ran during this check.
The first missing-PYTHONPATH attempt and the incomplete fixture failure are
retained alongside the final passing log.

Study100 remains frozen and retains its original registration and analysis.
Its declared artifact integrity verified under that instrument, but its
manual receipt did not preregister all implicitly loaded dependencies. It
must not be promoted to final runtime-identity acceptance. Future studies
must use a complete owner invocation before registration; this change does
not rewrite historical evidence.
