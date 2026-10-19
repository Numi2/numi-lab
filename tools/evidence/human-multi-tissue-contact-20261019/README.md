# Bounded multi-tissue contact residual qualification

This increment extends the existing Matter full-contact residual line search to an explicitly selected static FEM-only scene with 2–8 deformables. It keeps one shared Newton step per environment, rebuilds private contact geometry for each trial, and compares the same volume-plus-contact residual on both sides. Accepted history remains read-only during trials. Existing singleton admission is preserved. The optional selector remains NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT.

The extension rejects unsupported coupling rather than guessing: no MPM, rigid/support, vascular unknowns, Human attachments, mutable topology, mixed fields, or external FEM loads enter this bounded mode. This is not an all-body qualification. Default simulation ownership, physical mass, timesteps, tolerances, Newton/FGMRES budgets, and line-search CCD are unchanged. The implementation uses existing GPU state and trial arenas; no CPU per-step physics or new physical state is added.

## Source and ownership

Only matter/src/runtime.mm and four owning kernels in matter/src/metal/fem.metalinc change. source-mapping-proof.json reconstructs the exact tested Metal source from the published contact-only baseline plus those kernels. FGMRES and Cauchy kernels are unchanged.

The host routes pure multi-FEM through the bounded full-private line search, and excludes it from the volume-only loop. Earlier experimental admission accidentally selected the volume-only loop despite logging a full-contact mode: that inconsistent comparison caused the earlier strict rejection. The retained strict-interval270-status10 evidence in the prior contact publication belongs to that earlier routing and is not a successful full-contact test.

Geometry validation publishes per-object minimum-J without treating diagnostic FD scratch as an arbitrary object array. Its existing singleton FD caller retains the minimum-J write. Every production caller binds the new constant explicitly.

## Native contact evidence

The fixture uses the existing reduced right kidney FJ3147 (126 nodes, 369 tets) and spleen FJ2561 (139 nodes, 483 tets). The spleen is explicitly translated to form a controlled collision. These positions are not an anatomical registration. Material values are synthetic Neo-Hookean test values (density 1000 kg/m3, mu 50 kPa, lambda 500 kPa), not measured organ material calibration. Each current pose is supplied as an explicitly unstressed test reference, with SHA256 source identities and per-tet material assignments; this does not infer an anatomical unloaded state.

At 0.25 ms, 0.5 m/s approach, relative residual 1e-6, 12 Newton iterations and 64 FGMRES iterations:
- Four initial approach steps passed. The fourth had 44 cross-object history rows, 35 positive normal rows and 0.000449592 N s summed normal impulse (1.798 N average normal force).
- The object momentum changes opposed each other. Their net magnitude was 5.45256e-7 N s, or 0.1395% relative to the combined change norm. This is numerical evidence, not clinical or whole-body qualification.
- An injected late reject returned status 9, rolled back exactly, and accepted the retry.

The longer unchanged-knob comparison is retained, including failure:
- Quiet separated control passed; the moving-away control passed 32 steps (8 ms), with no cross-object rows. Both rollback/retry checks passed.
- Approach accepted six steps (1.5 ms), with positive contact force on steps 3–6, then rejected the seventh with status 10 / Newton iteration 10. Its last diagnostics were [9.56611e-4, 9.57181e-4, 1.01687e-7, 8]. No CCD failure witness was populated.
- The requested 32-step approach did not pass. Do not promote the short contact result into sustained deformable contact qualification.

## Compatibility and native scene

The singleton finite-difference diagnostic ran with one right-kidney FEM, contacts disabled, one fixed node and gravity. An unused scalar material state reserves the existing diagnostic scratch without entering the constitutive energy; compile maximum-rate exponent is explicitly zero. Minimum-J, residuals, operator action, centered difference and integrity fields are identical between runtime261/contact-only and runtime272/updated kernels. The relative operator-vs-FD errors remain 13.26% and 21.20%; matching the previous runtime establishes compatibility, not tangent accuracy. See fd-compatibility-comparison.json.

The complete retained native Human scene accepted one 0.25 ms step using the updated runtime and kernels. Its coupled physiology CSV is byte-identical to the prior contact-only native smoke. Invocation time was 54.631 s including initialization; this is not a sustained real-time-factor measurement. The retained layer configuration contains aorta/fascia/ACL, not all ten optional deformables. No visual verification was performed.

All native execution used SSH numi: Apple M6 Mac mini, Mac18,5, 16 GB, macOS 27.0 build 26A425. Exact source, asset, binary, library and Metal hashes, launch environments, compact receipts and logs accompany this file. Continuous long-horizon body/tissue mechanics, the failed seventh approach step, and all-tissue native qualification remain open.
