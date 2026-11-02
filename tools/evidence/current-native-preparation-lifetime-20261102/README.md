# Bound native receipt-import temporary lifetimes

The complete Skin/Fat scene reparses the large anatomy receipt for skin parent bindings, vessel parent bindings, and anatomy construction. Foundation receipt trees created under the process-wide pool can survive into Metal execution after all imported values have been copied. Three local autorelease pools now drain these temporary trees before GPU residency. The escaping state remains owned by the existing C++ containers and native owners.

No source geometry, material equation, force row, time integration, residual tolerance, or accepted-state policy changes. The execution owner and publication owner have different surrounding work; only the same nine-line insertion was applied to each. Full execution-source hashes and separate preimage/current hashes are retained.

Validation on SSH Numi: production native build; immutable shared storage and grouped-contact controls; native attachment force, deformation, rejection rollback and replay controls, with inactive and active contact; and the publication owner native application syntax check. All pass. Existing compiled Metal is reused; no shader source change is claimed.

A single large-scene GPU-entry pair now reports rounded vmmap footprint9.8G to9.0G, peak13.3G to13.1G and DefaultMalloc allocations10,338,485 to74,307, with29 preparation/initial-state outputs byte-identical. This does not establish sustained throughput, matched multi-run peak memory, convergence or accepted full-body motion. Trial106 stopped on its extra-swap guard. Trial115, after restart and storage recovery, stopped at 15_minute_diagnostic_budget before a terminal solver result. The matched trial retains source/mass checks, all 15,453,677 body contact rows, strict candidate overflow rejection, true-residual certification and original physical acceptance checks.

Apple documents local pools for reducing temporary-object footprint: [Using Autorelease Pool Blocks](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html). The retained one-pair result is limited to the sampled GPU-entry event; sustained benefit remains unqualified.
