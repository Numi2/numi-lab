# Box-plane certificate: where the thin-impact search stops

The existing unpromoted four-contact contact shader and two diagnostic-only
shader variants were run on the same 12 hybrid fixtures and the production
16-iteration limit, with both experimental selectors disabled. Each variant
matched its fresh control in all reported physical and transaction fields
(excluding elapsed time), and in 96 captured event records except the
intentionally changed unresolved-pair diagnostic fields. This comparison
covers the fixture-exposed state; it is not proof of full owner restoration.

For the two 250 rad/s thin-impact cases, the first impact event was followed
by an unresolved swept cast. The per-corner subdivision loop exhausted its
16-iteration budget on corner 0; it did not stop because its stack was full
or its midpoint was unrepresentable. The last processed node was certified
positive, yet the stack still held one interval at scale 1 and five at scale
0.5. Its normalized lower bound was 0.25 and 0.0234375 respectively. Later
trace passes repeat the already failed state and are not independent events.
When the final node was positive, the alternate negative-witness and bracket
conditions were not evaluated; analysis reports those fields as null.

The unexamined intervals might be clear or might contain an interaction. This
capture does not prove either, establish convergence with a higher budget,
or justify accepting the step. It does not repair the separate contact-row
residual failure or qualify the four-contact candidate, which also fails
mixed rod-rigid response. A useful next step is to inspect and independently
bound the pending intervals and the source of loose subdivision bounds,
then test any repair at cap 16 with full-step impacts and near misses.

The first analysis invocation raced the still-running probe and failed JSON
parsing on an incomplete output; complete outputs were read only after the
process exited and then analyzed successfully. The initial plan creation also
had a shell quoting error; the partial plan is retained. Source, build
commands, input and output hashes, baseline and instrumented traces, scripts,
and the diagnostic patches are retained. These builds are research-only;
no production shader was modified or promoted. Timings are not a benchmark.

The retained build commands reference existing local AIR objects and the
candidate binary; their hashes are recorded, but those binary dependencies are
not included in this evidence directory. Rebuilding on another host requires
reproducing those inputs or obtaining matching binaries. The original
Collision source is the file at main commit 2781fcf44; the instrument scripts
apply to a local copy named Collision.before.metal.
