# Graph Schwarz completed-cycle guard overlay for frozen source 067

This is an isolated proposed overlay against
`/Users/home/cardboard-implementation-20261014/snapshots/source-consistent-armijo-alpha-and-topology-capture-067`.
It does not modify that snapshot or the shared working checkout.

## Audited behavior

The five selector-5 graph Schwarz kernels are listed in the common pipeline
array at `matter/src/runtime.mm:3690-3694`. The runtime builds graph matrices
and factors them only for selector 5, around `runtime.mm:14623-14708`.
That work occurs before the per-restart `nm_fgmres_begin` call at
`runtime.mm:15109`, so its completion predicate is not yet authoritative and
this overlay deliberately does not skip graph setup or factorization.

The local patch solve runs after `nm_fgmres_begin`, in the selector-5 branch at
`runtime.mm:15527-15575`. The existing graph apply kernel already zeros its
`preconditioned` output when `states[environment].diagnostics.z > 0.5`.
However, `nm_fgmres_graph_solve_patches` still performs per-patch triangular
solves for those completed cycles. The overlay binds the current FGMRES state
at Metal buffer 17; on the same uniform completion predicate, each group
zeros its own `patchCorrections` entries and returns before the local solve.
The physical operator, status/certificate path, convergence tests, and graph
factorization are unchanged.

Graph Schwarz is selected by the immutable compiled-world value
`executionBudgets.z == 5`. The runtime pipeline-creation loop now skips only
the five graph Schwarz pipelines when the compiled selector is not 5. Selector
5 still creates the full set, and its existing guarded dispatch sites remain
the only consumers. This is pipeline-state creation gating; the Metal library
image is still loaded normally.

## Review and limits

The patch is bound to the exact frozen-067 source hashes in
`source-overlay.json`. Static review confirmed one runtime dispatch site for
the graph solve and apply, and that both are inside the selector-5 branch. The
completed flag is read after `nm_fgmres_begin`; the graph setup/factorization
before that point remains unconditional within selector 5.

No build, native run, or GPU execution was performed for this overlay. It is
not performance or numerical qualification. The root task must integrate it
into a fresh source snapshot, compile it, and compare full state/certificate
history against the unguarded selector-5 path before adopting it.

Files `runtime.mm` and `fgmres.metalinc` are the complete overlaid source files;
the `.patch` files are their diffs against frozen 067.
