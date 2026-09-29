# Native torso anatomy pose snapshot

When `numilab_human_myosim_visual_probe` renders an NHANAT1 torso anatomy
payload, it now writes a `.torso-anatomy-poses.json` companion to its MRVPACK2.
This inspection-only receipt contains the exact `bodies` vector passed to
`makeMotion` and the native renderer, restricted to the surface owners. It
records the registration fingerprint, surface count, Core body IDs, world COM
positions and world inertial orientations in xyzw order. Duplicate surface
owners are emitted once. Output completion failures stop the inspection run.

The pack retains its original local geometry, identities, transforms and
symbolic articulated bindings. This change adds no physics update, force,
render transform, alternate pose authority or runtime ABI. An independent
oracle can decode the pack and use the exported native poses to compare every
source vertex and normal with the pinned MuJoCo source inertial frames.

The thin inspection executable was compiled with warnings as errors and linked
against the unchanged retained libmetalrobo runtime. Apple M4 captures cover
raw source rest, equality-projected neutral and coupled torso motion, plus
1024-pixel neutral and posed board captures. The Human focused verification
passes 29 tests, including wrong owner, topology, displaced vertex, normal,
pose and source hash corruption. All 22 selected surfaces and 92,623 vertices
match source geometry. High-resolution posed maximum error is 0.169
micrometres at the existing 20-micrometre geometry serialization bound.

Evidence is owned by the Human repository in
`Docs/TORSO_ORGAN_COVERAGE_20260929.md` and
`Build/torso-organ-coverage-20260929`. Native binary SHA-256 is
`e8e447b24636e8a3f0c8c8c641e727cbc4fec3bddc64abb60454a84f486a2bec`;
runtime SHA-256 is
`6b9062e84bdacd6b86b413d5c549cd0046d9a271fb0caa141e03db41ffa213f6`.
Native base revision is `d1a13a6b48675743fc242d7b7b7f56e2e40b4209`.

This is source and visual-articulation evidence, not complete organ anatomy,
deformable organ mechanics, clinical validation or a standing rollout.
