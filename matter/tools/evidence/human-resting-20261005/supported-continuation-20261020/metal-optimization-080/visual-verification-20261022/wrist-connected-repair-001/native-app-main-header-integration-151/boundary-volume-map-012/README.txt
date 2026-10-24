Explicit skin volume contact mapping preparation, 2026-10-24

Bed samples and skin-bone O/D samples now use the exact boundary node map.
Material-column adjacency accepts optional explicit per-volume-node columns:
the matched O/I/D roles share an ID; actual interior volume nodes have ID zero.
There must be exactly three boundary nodes per column. No contact authority,
material equation or physical acceptance tolerance is removed.

Validation: strict portable C++ tests pass for legacy tags/fingerprint parity,
unpaired and reordered rows, untagged interior nodes, and failed-preflight atomicity.
Full native app compiles/links. Source-only baseline positions/masses, detailed skin
and body poses are byte-identical, as are contact/compiled fingerprints.
No physical step or repaired-geometry admission is claimed. Generic volume owner
and source geometry validation remain pending. Legacy optional full-bank comparison
rejects unpaired row labels rather than emitting falsely classified rows; mapped
initial-node capture supports arbitrary nodes (interior rows use layer 3).
