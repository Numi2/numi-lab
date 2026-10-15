# Deterministic contact-node incidence sort (source 076 overlay)

This external overlay adds an opt-in, deterministic builder for the node-to-
deformable-contact-corner incidence CSR. It is intended to replace the legacy
per-node scan of every active contact when that scan becomes the limiting
operation. The legacy kernels and default behavior remain intact.

The only production source changes are `runtime.mm` and
`metal/contact.metalinc`. Set `NM_CONTACT_NODE_INCIDENCE_SORT=1` before
`Runtime::initialize` to select the new builder. If the variable is present
with any other value, initialization fails closed. With the variable absent,
the existing count/scan/scatter pipeline is used. The selected builder is used
for both the production incidence list and the serialized private full-contact
Armijo incidence rebuild. Its key and payload arenas are reusable scratch; the
builder does not write contact histories, accepted state, or solver
certificates.

For each of the six corners in each active-list position, the emitter writes
an integer key `(nodeID << 32) | (activePosition * 8 + localCorner)` and a
payload `contactSlot * 8 + localCorner`. Invalid, inactive, out-of-range, and
padded records receive the all-ones key. A fixed ascending bitonic network
sorts the keys independently per environment. The `activePosition` part of
the key preserves the legacy summation order even if the active list repeats
a contact slot; the payload preserves the original slot and corner identity.
Two lower-bound searches per node create the count, the unchanged prefix scan
sets `first`, and the sorted payloads fill `nodeIncidence`. Only entries inside
each range are consumed. The native checker also compares untouched capacity
entries and range metadata against the legacy output.

The workspace uses checked 64-bit arithmetic before allocation. It checks
logical and padded record counts, `uint32` kernel indices and payloads,
`size_t` byte extents, each Metal buffer's `maxBufferLength`, and resident-byte
accounting. It rejects unsupported or overflowing inputs rather than
truncating records. For contact capacity `K` and environment count `E`,
`S = nextPow2(6*K)` records are allocated per environment: `8*E*S` key bytes
and `4*E*S` payload bytes. At `K=131072`, `S=1048576` and this is 12 MiB per
environment (24 MiB for two; 48 MiB for four).

The bitonic network has 210 stage dispatches per incidence build at that
capacity, independent of active count; it executes 110,100,480 compare pairs
per environment. Each node also performs two 20-step binary searches. This
removes the legacy `O(N*K)` node/corner scan, but it may be dispatch-bound and
can lose when the active list is small. No performance claim is made here; the
native prepared-input checker verifies GPU output parity, and a later
source-bound runtime pair must measure time and complete-state/history
invariance before the selector is recommended for use.

## Contents and base binding

- `patches/contact-node-incidence-corner-sort.patch` contains only the
  production changes relative to frozen source 076.
- `tools/cardboard_contact_node_incidence_sort_reference.hpp` provides the
  CPU legacy and sorted builders plus checked workspace sizing.
- `tools/cardboard_contact_node_incidence_sort_check.cpp` is the portable
  exhaustive/reference test.
- `tools/cardboard_contact_node_incidence_sort_native_check.mm` dispatches
  both the new and legacy Metal pipelines on the same prepared input and
  compares complete ranges and incidence storage to each other and to the CPU
  legacy reference.
- `SCALING_ADDENDUM.md` records the first asymptotic bottleneck found without
  modifying the prior sealed inventory.

The production patch is based on frozen source directory
`/Users/home/cardboard-implementation-20261014/snapshots/source-stable-free-graph-and-general-arithmetic-076`
with snapshot SHA-256
`9b9047688d4da51739f97a35c0c0c24c57a38bcf4631088d4377ab9697c1d516`.
The base `runtime.mm` SHA-256 is
`5c70789647a9c0ceacaeb10565aaaee30e2a901761c1d04e6e431fdafe0aeef1`; the
base `metal/contact.metalinc` SHA-256 is
`69c3ec930d0b382dfe278fa42c56353782438ded77d8224a3ccfd6989e3067c9`.
The overlay does not change ABI or CMake.

## Checks run

- CPU reference check: `49162` assertions passed, including all 4096
  one-slot corner-support assignments over active/inactive contact, invalid
  active slot, and active count `0/1/2`; explicit two-environment cases cover
  repeated active slots, duplicate corners, invalid IDs, and clamped counts.
- `runtime.mm` Objective-C++ syntax check passed against source 076 headers.
- Native checker Objective-C++ syntax check passed with a placeholder
  `NUMI_MATTER_METALLIB` compile definition.
- Metal shader compile passed with two existing warnings in source-076
  graph-Schwarz threadgroup variables (`sharedValid`, `refinementValid`);
  there were no errors in the new incidence kernels.
- The native checker has **not** been executed on GPU. No physical steps were
  run. Integration, native parity, complete transaction-history invariance,
  and runtime are pending root's sealed build/run.

## Integration guidance for root

Apply the patch to a clean source-076-derived tree, then copy the three
`tools/` files into `matter/tools/`. Add a CPU test target for
`cardboard_contact_node_incidence_sort_check.cpp`. Add a native target for
`cardboard_contact_node_incidence_sort_native_check.mm`, define
`NUMI_MATTER_METALLIB` to the built `NumiMatter.metallib`, link the existing
Matter/Metal/Foundation dependencies, and make it depend on the shader target.
The native checker is prepared-input kernel parity only; it does not test
`Runtime::initialize`'s environment parser, transaction rollback, or
production/private call-site selection. For runtime qualification, run the
same sealed full-contact fixture twice with only
`NM_CONTACT_NODE_INCIDENCE_SORT` differing, compare every accepted/rejected
state and history payload, and report the selector's recorded workspace
size. Keep the selector opt-in until those checks pass.
