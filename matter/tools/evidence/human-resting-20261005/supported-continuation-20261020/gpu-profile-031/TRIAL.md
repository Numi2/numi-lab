# Bounded full-scene GPU timing instrument 031

Run only after native030 and with no concurrent GPU job. Use identical assets,
source67c85cb3 and physics configuration for two125us steps per replay.

The existing borrowed-command-buffer instrument has2048 timing-record slots.
A retained older profile gives1346 records at interval64; doubling that for the
current solver budget would exceed capacity. Use two existing configurations:
- template.json: all kernels, root1 only, every256th dispatch. This samples the
  frequent solve kernels but misses kernels called fewer than256 times.
- contact-template.json: every dispatch at root1 for sort_surface_primitives,
  compact_deformable and scan_deformable_node_incidence only. These three are
  selected from the older timing evidence, not asserted current bottlenecks.
These are sequential instrumentation replays, not parallel physical owners.

Require the first two accepted CSV rows of each to be byte-identical to
native030 and to each other. Retain dispatch counts, all available timestamps
and dropped/unavailable diagnostics. A dropped sample makes coverage censored.
Sampling is descriptive; do not infer a whole-scene speedup or complete steady
runtime profile from these two roots. Choose further work by measured cost,
not geometry size. No physical readbacks, equations, timestep or state
ownership changes. This is not supported-body or physiological qualification.


## Captured outcome

Both sequential profiles exited 0 and emitted two accepted CSV rows. Their first
two data rows are byte-identical to native-030 and to each other; see
native-integrated/row-parity.json. The all-kernel sparse capture retained 630
elapsed-time samples and 224 dispatch-count records. The selected-contact
capture retained 316 elapsed-time samples and six dispatch-count records.
Neither log emits an explicit dropped-sample counter, so drop count is unknown.

The sparse and contact filters differ, the root filter retains only root 1, and
each run is only two physical steps. Their roughly 27.9-second wall times include
startup and are not used for cost ranking or speedup claims. Exact invocation
and pre-run provenance are retained. The recorded profile binary, Metal library,
and source-file hashes match the native-030 source-67 baseline; no later
in-flight source edits are attributed to these profiles.
