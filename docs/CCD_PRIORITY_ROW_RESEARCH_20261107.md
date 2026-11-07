# Bounded priority-row contact experiment (failed)

This is an unpromoted research candidate. Each distributed island performs at most eight priority block pivots per configured solver sweep, against the actual articulated and scene-body response, and the existing final normal complementarity certificate is retained. The non-distributed path has the separate all-path containment certificate. Friction uses the existing conditional tangent and cone helper; these checks do not establish general friction or mixed-constraint correctness.

Independent FP64 normal-only examples solve a zero-lever owner-shaped contact and a set of identical rows in one coordinate update, and three mutually orthogonal contacts in three updates. They also demonstrate that a correlated two-row case has residual sequence 1, 0.5, 0.25, 0.125 under four pivots: a fixed cap is not a convergence guarantee.

The real 257-contact owner-shaped impact completes at the production two-sweep budget, with normal impulse 0.081749998 N s, momentum error 2.8e-12 N s, and contact kinetic work -0.003341531 J, consistent with the independent solution. Identical replay was observed. The 33-contact owner-shaped impact correctly rejects via non-distributed containment and remains unresolved.

The prior physical 72-case matrix regressed from 72 accepted to 61. At two sweeps the candidate rejects all three previously accepted 257-contact incoming families: aligned frictionless, mixed directions, and aligned frictional. Two 33-contact one-sweep cases are also rejected by the added certificate. Therefore the owner success does not pass meaningful-motion qualification. Rejected cases restore initial q, v, scene state and empty contact cache; all 72 replay identically. Force-free separation passes in the matrix. This is fixture-specific evidence, not a general rollback or friction certification.

In one exploratory matched run the completed owner candidate used 65.5 ms GPU time, while the previous candidate's rejected owner step used 6.4 ms. These have different acceptance outcomes and are not a performance comparison. Timing samples for all cases are retained as diagnostic evidence; no speed claim follows. The prior candidate's passing matrix is unchanged and independently retained on its branch.

[Raw results, source patch, FP64 reference and source/binary identities](evidence/ccd-priority-row-evidence-20261107.tar.gz) have archive SHA-256 1b83e6deb01df8d63e573c876016e52adb9dbc7853382aaf93165de2e8c36902. Metallibs and executables remain host-only; their SHA-256 identities are inside the archive.
