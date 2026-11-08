These deterministic zero-policy contracts predate the object-track output
stride correction. Both contain object tracks; `0.npolicy` uses compact actor
history and `1.npolicy` has an extended masked-depth suffix. Both now reject
because the subsequent object-track depth-validity repair changes semantics
for every object-track policy. Rebinding is tested; these are not trained
policies. The newer `object_track_fixtures` include non-track controls.
