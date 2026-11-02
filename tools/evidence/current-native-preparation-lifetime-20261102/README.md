# Bound native receipt-import temporary lifetimes

The complete Skin/Fat scene reparses the large anatomy receipt for skin parent bindings, vessel parent bindings, and anatomy construction. Foundation receipt trees created under the process-wide pool can survive into Metal execution after all imported values have been copied. Three local autorelease pools now drain these temporary trees before GPU residency. The escaping state remains owned by the existing C++ containers and native owners.

No source geometry, material equation, force row, time integration, residual tolerance, or accepted-state policy changes. The execution owner and publication owner have different surrounding work; only the same nine-line insertion was applied to each. Full execution-source hashes and separate preimage/current hashes are retained.

Validation on SSH Numi: production native build; immutable shared storage and grouped-contact controls; native attachment force, deformation, rejection rollback and replay controls, with inactive and active contact; and the publication owner native application syntax check. All pass. Existing compiled Metal is reused; no shader source change is claimed.

The large full-Human trial is pending. This commit does not claim reduced peak memory, accepted full-body motion, convergence, or a performance gain. Earlier native trials stopped under resource guards without a terminal solver result. The matched trial retains source/mass checks, all 15,453,677 body contact rows, strict candidate overflow rejection, true-residual certification and original physical acceptance checks.

Apple documents local pools for reducing temporary-object footprint: [Using Autorelease Pool Blocks](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html). The actual large-scene benefit still requires measurement.
