# Connected skin/fat conditioning and conservative physical transfer

The former root-20 stall reached the adipose validity bound J=2.5 in original
source cell 1170724 (native element 1078892). Its normalized determinant was
1.2930958e-5 and minimum altitude approximately 29.8 nm. Exact two-cell and
four-cell swaps failed; a connected seven-to-five replacement raises its
minimum normalized determinant to 0.0159062 without moving vertices.

The existing cavity NPZ/importer now admits bounded connected replacements
with exact Float32 orientation/volume, closed connected cavity checks,
non-overlap certification, complete source ancestry and unchanged material,
donor and regional identities. The existing compensated conservative flow
solver routes restricted source transfers during initialization. No simulation
state, force owner, material law, clearance, physical timestep or acceptance
tolerance is replaced.

A batch proposed 500 replacements. One failed the native closed-sphere check.
Of the remaining 499, source routing resolved 11 transport-radius rejections;
31 genuinely infeasible proposals retain their baseline cells. The final 468
connected replacements plus 41,391 retained legacy cavities pass native
geometry and physical mass admission. Total skin/fat cells: 1,453,640 to
1,453,060. Geometry and mass admission do not establish motion validity.

Native135 accepts **24** steps at 15.625 us, then rejects root 24 with status 10.
The prior geometry accepts 20 steps. This is only **0.375 ms** of initialization,
not settled rest, a breathing cycle, sustained throughput, or 0.1x qualification.
See native135-summary.json for actual timing and the remaining failure.

Validation: existing 3-to-2 controls; connected seven-to-five conservative
mass/provenance controls; malformed, inverted, capacity and transport cases;
restricted-flow feasibility and fail-closed tests; all 41,859 cavities checked
by the existing geometry executable; exact construction replay; 39/39 execution
source hashes independently reconstructed.

The geometry-only checker rejects bad proposals before costly scene preparation:

    numi-human-skin-fat-fixed-cavity-check --geometry-proposal SOURCE.npz PROPOSAL.npz

It deliberately does not certify mass or physical motion.

Exact execution is the retained dirty f75955f0 checkout plus the source overlays
listed in source-recovery-135.json; the published native entrypoint receives
only the two new report fields, preserving unrelated source differences.
Earlier failed trials and their source/input pins remain evidence. Assets and
reconstruction are published in Numi2/numilab-human under
tools/evidence/connected-skin-fat-cavity-conditioning-20261104.
This is a mixed-source reference assembly, not measurements of one person.
