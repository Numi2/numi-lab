# Stateful-check selector-5 diagnostic update

This overlay corrects one stale assertion in the source-065 stateful package round-trip test. The test mutates `mixedSolver.executionBudgets.z` to 5. In source 065, selector 5 is a defined graph-Schwarz preconditioner; the generic mixed-solver policy accepts the defined value, then the dedicated graph-Schwarz scope validator rejects this fixture because its compiled world contains mixed MPM/FEM ownership. The validator's scope rejection is the correct fail-closed behavior.

The updated test expects the graph-Schwarz scope error for selector 5 and adds selector 6 as the out-of-range case that must still fail with `mixed solver policy`. No validator, compiler, physics, or package code changes are included.

Base: `/Users/home/cardboard-implementation-20261014/snapshots/source-angular-and-graph-schwarz-integrated-065`.
Base `matter/tools/stateful_check.cpp` SHA-256: `aae7021ce00ac272eac1c7a5ed3893b266d407e3df7fa5aceb0d9c24d01836af`.

Verification: compiled the overlay file with the source-065 `compile_commands.json` flags, linked it against the untouched source-065 `libnumi_matter_compiler.a`, and ran the resulting checker successfully. Output schema was `numi.matter.stateful-compiler.v1`, exit code 0. This is a CPU compiler/package validation check only; it does not qualify selector-5 numerical behavior.
