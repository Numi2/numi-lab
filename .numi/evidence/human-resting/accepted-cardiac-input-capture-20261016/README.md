# Accepted cardiac geometry inputs

The existing opt-in accepted geometry capture now retains the exact bound cardiac, wall and common-coordinate GPU buffers beside each selected MRVPACK2 frame. The existing receipt schema is extended with their paths, SHA256 hashes, ABI spans and record types. No simulation state, update scheduling or physical owner changes.

Native regression ran on SSH Numi with the frozen runtime and one accepted 0.002-second step. Vertex, index and primitive sections, body and respiration states, and the coupled physiology trace are byte-identical to the retained control. The seven captured common coordinates equal the Float32 round trip of every matching completed-frame trace row. In common mode the inactive legacy cardiac q buffer is intentionally NaN; it is captured without interpreting it as active state.

The two initial rows at step/time zero have matching cardiac coordinates but different body audit values. This result authenticates the captured coordinates; it does not establish unique step/time selection for arbitrary trace fields. The diagnostic is retained.

The build report records exact source hashes, compiler/linker commands, ABI headers and binary identity. The native launch and comparison scripts remain at the paths and hashes in the reports under /Users/n/numi-human-retained-delivery-20261009. The original failed comparison assumptions and sparse-build attempt are retained there. The source revision is the pre-change base; the modified visual header is identified by SHA256.

This is an evidence-export regression. It does not qualify anatomy, full pose replay, rejected-step rollback, sustained physiology or performance. Long-run anatomy qualification remains pending.
