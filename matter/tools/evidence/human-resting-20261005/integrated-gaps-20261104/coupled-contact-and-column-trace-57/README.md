# Coupled contact blocks and late-column evidence (57)

This is a bounded experimental preconditioner extension, default off. It does
not qualify the complete Human, supported rest, sustained physiology or 0.1x.

The existing eight-node thin-cell Schwarz patch now has an opt-in complete
principal PSD contact block across nodes already in that patch, instead of
only each node's 3x3 block. It uses the existing six-corner incidence, material
patch matrices and factorization. Fixed columns, mixed timestep chain rules,
mollifier curvature and duplicate-corner sums are retained. No persistent
buffer, contact exclusion, physical force or acceptance tolerance was added.
NM_FGMRES_GRAPH_COUPLED_CONTACT_BLOCKS=1 requires the admitted thin-cell
configuration and NM_FGMRES_GRAPH_FULL_PSD_CONTACT_BLOCKS=1. Sidecar
respiratory/Cardiac runtimes stay excluded. Function constant 42 defaults off.

The production-kernel oracle passes 86,544 checks (23 cases, two environments),
including full eight-node patches, six corners, duplicate nodes, off-patch
nodes, fixed columns, rigid owner filtering, mixed rates, mollifier curvature,
empty work, malformed rows/capacity, nonfinite rejection, completed-tail reuse
and undefined-versus-disabled parity. Maximum matrix-entry error against the
independent double reference is 8.85949e-7; random-direction product error is
1.4973e-6. The solver regression suite passes 2,671,801 checks.

## Integrated result: not promoted

Trial 297 uses all fine coordinates and unchanged physics. The graph contains
8,290 patches covering 30,768 of 318,704 free nodes (9.65%). It accepts 22
15.625-us roots, then fails the unchanged nonlinear certificate at root 22:
1.7993234e-5 versus 1e-5. The failure receipt retains accepted common
coordinates and accepted Brain time 344 us while the rejected candidate is
359 us. This is insufficient for a whole-body rollback claim by itself.
There is no valid complete-horizon speedup or force/strain qualification for
the experimental preconditioner.

## Reference trace: exact state preserved

Trial 298 disables the experimental graph and adds the existing bounded
column observer at root 28 plus a fresh true residual at Newton iteration 15.
All 32 roots accept. Its 318,974 node and 1,479,479 tet endpoint records and
numerical traces match 290 exactly. Body time is 230.862449 s, versus
230.856208 s in 290. This is an observational comparison, not a new speedup.
The duration is 0.5 ms of release initialization, not supported rest.

All 256 root-28 columns execute. Newton 0-5 reduces the scaled residual to
roughly 0.89% of its initial value in 16 columns; Newton 6-15 leaves 28-29%.
At Newton 15 the scaled Arnoldi norm is 0.01798455 from 0.06232814, while the
fresh physical norm is 5.38055e-6 from a 2.85364e-5 physical RHS. These are
different norms and must not be compared as an Arnoldi residual-gap error.

Deformable contact shrinks alpha from 1 to 0.0003464354 at the final Newton
iteration. The per-object alpha changes at after_deformable_limit; the
environment scalar publishes it later at after_support_limit. This naming
does not implicate support physics. Most of the computed correction is
therefore discarded at this late iteration. Raising budgets or merely
loosening the existing forcing cap is not established as useful.

Preserved raw stderr, exact plans, receipts, source/binary hashes and the
compact comparison distinguish these results from earlier mesh/revision
trials. Initialization, native presentation and five-minute qualification
remain incomplete. No images were opened and no geometry was changed.
