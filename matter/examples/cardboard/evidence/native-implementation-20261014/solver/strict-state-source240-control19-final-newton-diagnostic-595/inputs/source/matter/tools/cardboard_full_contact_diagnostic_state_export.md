# Full-contact diagnostic state export

The native `cardboard-full-contact-armijo-check` accepts
`--diagnostic-state-output DIR` to write a deterministic before/after snapshot
pair for each attempted transaction. `DIR` must be a new, unused directory.
Files use `transaction-NNNNNN-before.bin`, `transaction-NNNNNN-after.bin`, and
`transaction-NNNNNN.json`; names contain no run timestamp or source path.

The binary files contain the raw payload produced by
`shipping_snapshot_detail::writeSnapshotPayload`, followed by the
`NMRAPDI1` extension for optional regional-Galerkin diagnostics. The payload
contains the complete `RuntimeStateSnapshot` fields, including material,
cohesive, rigid, and contact history vectors. It is a diagnostic serialization,
not an `.nmcb` checkpoint: the fixture does not call accepted-checkpoint
validation or publication, and a rejected after-state is retained with
`accepted: false` in its JSON record. If a transaction fails before an after
snapshot can be read, the record states `after_present: false` and preserves
the before payload plus the bounded completion stage.

Each JSON record reports requested and observed control cursors, per-vector
counts, all available environment statuses and solver certificates, and the
derived acceptance bit. Non-finite status or certificate numbers are emitted
as JSON `null`; the binary snapshot preserves their original bits. Compare
these exports only between runs with identical compiled inputs and physical
arguments, changing only the diagnostic selector under test. Matching raw
payloads support state-invariance review; they do not establish that a failed
transaction converged or qualify physical cardboard behavior.

`accepted_state_vector_sha256` additionally hashes the exact bytes of each
listed persistent material, topology, FEM/particle, vascular, rigid-body and
contact-history vector. For a rejected transaction, compare these hashes,
vector counts, and the control/physics cursors with its before state. Status,
certificate, contact-sample and regional-linearization scratch are separate:
the complete before/after payloads need not match after rejection. The hash
list is an explicit rollback scope, not proof that every runtime scratch
arena was restored.

`cardboard_full_contact_history_compare.py` verifies the sealed source/build,
physical argv and diagnostic-only environment differences before comparing
the raw files. It reports output parity and persistent-state rollback
separately. Cases 063/064 demonstrate why: the diagnostic on/off files match
byte for byte while both failed attempts retain a changed BOX projection.
Neither parity nor the comparison command's parity exit status means that
the rollback or physical solve passed.

Source44 comparisons078/079 and081/079 additionally verify all18 persistent
vectors and the accepted cursors are identical before/after rejection.
Comparison081 selects the actual Newton16 failure contact action capture;
078's selected Newton20 is after the failure. Both retain identical raw
production files against079. The nonlinear transaction itself still fails.
