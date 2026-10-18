# Blood MPM handoff increment (isolated source package)

This package carries the MPM receiver handoff source and a native accepted/rejected Runtime callback fixture.

The shader activates pre-authored dormant MPM receiver capacity from externalBlood.pendingParcel, updates represented blood/gas as a subset of the existing external inventory, and relies on Matter's accepted/candidate/checkpoint arenas for rollback. The 208 Runtime finalizer's MPM-only rollback path is supplied by the coordinated contact-runtime patch; it is not duplicated here.

The native test seeds an external inventory directly. It proves receiver activation, mass and O2/CO2 ledger representation, injected rejection rollback, same-context retry, equality to an uninterrupted accepted control for particle/material/vascular/vascular-clock/scheduler/control-cursor snapshots, and exact snapshot restore. It is a transaction unit only. It does not prove live CVSim-outlet wiring, whole-body circulation-to-MPM integration, a resolved free-surface liquid model, or clinical blood rheology.

The fixture uses a GPU-only owner-status preparation kernel at each attempt boundary. This models MetalWorld's preparation of a fresh world-status record after a rejected attempt; it does not modify Matter particle or physiology state on the CPU. Retry uses the same controlStep as the accepted control.

Build and run scripts retain the exact Numi paths, source/runtime hashes, compiler/link argv, and exclusive native wrapper. The build JSON is the pin authority for successful attempt 005. Attempts 002 and 003 preserve the earlier retry failures; 004 passed before explicit status/count fields were added; 005 is the final measured pass.
