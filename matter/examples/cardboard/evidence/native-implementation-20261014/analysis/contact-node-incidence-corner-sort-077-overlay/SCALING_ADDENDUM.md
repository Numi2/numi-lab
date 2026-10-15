# Scaling addendum: contact-node incidence construction

This is an addendum for the source-076 audit, not an edit to its sealed
inventory. It identifies an algorithmic scaling cost in the current
deformable-contact CSR builder; it does not claim that the new implementation
has passed a native runtime or performance qualification.

In `matter/src/metal/contact.metalinc`, the legacy
`nm_contact_count_deformable_node_incidence` and
`nm_contact_scatter_deformable_node_incidence` each run once per unified
continuum node. For every node, both kernels walk every active contact and all
six corners. With `N` nodes and `A` active contacts, they do up to
`12*N*A` node-equality checks per environment across count and scatter. This
is real repeated work, not just oversized allocation. `runtime.mm` dispatches
the count, scan, and scatter stages from the incidence builder in both normal
contact processing and private full-contact Armijo candidate reevaluation.

For the audited envelope `N=528444`, `K=131072`, one environment, and worst
case `A=K`, the two legacy passes perform up to `831170543616` corner/node
comparisons. The opt-in sort instead emits at most `6*K` records, sorts a
padded array, performs two binary searches per node, and scatters the valid
payloads. For these dimensions the padded sort has `S=1048576`, 210
bitonic-stage dispatches, `110100480` compare pairs, and `21137760` binary
search comparisons; the two sort arrays occupy 12 MiB. The cost scales by the
environment count in storage and compared records.

This is an operation-count estimate from source and dimensions, not a measured
speedup. Bitonic dispatch overhead and low active counts can erase the
asymptotic advantage. The implementation therefore requires an explicit
environment opt-in and keeps the legacy route as the default. It must remain
unqualified until the native prepared-input parity check and sealed on/off
runtime comparison are run. No contact capacity, pair-generation rule,
physical operator, acceptance threshold, or rollback behavior is changed.
