These deterministic zero-policy contracts predate the visual depth-validity
change. `0.npolicy` has no visual depth suffix and `1.npolicy` contains eight
masked-depth values; both include Franka object tracks. Both now reject because
the later object-track frame/depth validity repair changes their observation
semantics. Rebinding is tested. The newer `object_track_fixtures` include
non-track controls. These are not trained policies.
If a containing branch changed the task fingerprint, the test first verifies
that the original pack rejects, then adjusts only its task binding in memory
while retaining its old observation fingerprint for the migration test.
