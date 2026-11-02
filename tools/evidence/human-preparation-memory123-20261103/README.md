# Human preparation and bounded upload memory

These changes preserve the physical update, source/geometry/mass admission,
accept/reject processing, and replay ownership. They reduce temporary CPU
preparation copies and cap retained CPU-to-GPU upload staging at 32 MiB.
The successful exact-intersection cache contains only exact input digests
and tested-pair counts; all other source/closure/topology checks still run.

Verification on SSH Numi (Apple M6, 16 GiB):

- Native build and existing skin-volume/replacement checks pass.
- Source-domain reduction-label corruption checks pass.
- NPZ regression covers stored/deflated parity, exact values, allocation
  reuse, CRC, required members, truncation, and payload-shape rejection.
- Existing native FEM attachment/rollback control passes with default and
  forced 256-byte staging, for inactive and active contact. Forced small
  staging executes 1,275 batches; rejected nodes/cursors/history/replay
  remain bitwise unchanged. This is a coupon, not full-scene throughput.

Full scene trials remain unqualified. Trial 001 stopped on 3 GiB additional
swap growth; trial 002 stopped on the 12 GiB disk reserve; trial 003 stopped
at the disk preflight. None reached an accepted physical step. The 0.1x
integrated target is not achieved or measured by these tests.

A separate idle macOS VTDecoderXPCService had a 53.4 GiB footprint.
After termination macOS attributed it to kernel_task and retained 22 GiB
swap. The process/VM evidence is retained. This does not establish the
simulator's GPU working-set requirement.

Historical assets are retained in the private recovery release
Numi2/macmini-evidence-archive, tag
numi-20261103-historical-physics-evidence-memory123. Each retired batch was
independently downloaded, checked member-by-member and restore tested.
Current inputs and transitive mass proofs are excluded. Final net disk
recovery is still being measured; the compression receipt is an
intermediate observation rather than a claim that 20 GiB has been freed.

A second increment adds opt-in immutable/state/contact/Krylov/proof allocation
phase counters and releases particle/FEM material-state upload defaults after
their last upload. The same four native controls pass after this change.
