# Slider support reaction from the rigid owner: staged

A dedicated two-body, one-axis ABA kernel variant records the reduced spatial
wrench propagated to its fixed parent, in world coordinates about that parent's
origin. It writes only after the ABA solve succeeds. MetalWorld is staged to
select that variant for its explicitly admitted slider and lend its private
scratch to the post-commit callback. The ordinary ABA variants are unchanged.
The Matter coupled execution guard remains closed.

An isolated native GPU test passed four cases: unloaded gravity, two distinct
axial forces and applied torques, and a transverse-force diagnostic with an
offset lever arm. The check independently evaluated acceleration and support
moment. This test calls the actual dedicated ABA variant; it does not involve
Matter, the complete MetalWorld command graph, an accepted coupled step, or
rollback. ABA source SHA-256:
`cbc6a44d962bd036a09548af71ac9b73b3ab7b4217e285ec0807a0905e2c46c4`;
isolated library SHA-256:
`4258cdc86a953702a97778bcc12c35a4110bc67c287648925ee57ca387b3224a`;
test executable SHA-256:
`6a826e6693f18ea110a6a7f28ccf74511563ce37a38fc0b70898ae9469a29408`.

The MetalWorld host and slider owner fixture passed syntax checks. A new
owner-only transaction case was staged to check applied moment and rejected
transverse force against the complete rebuilt owner; the later result is below. The
borrowed reaction is scratch, is meaningful only with successful ABA status,
and must not be treated as an accepted physical state or retained across the
callback. Whole-owner rollback and the native FEM/slider gates remain open.

The rebuilt complete MetalWorld owner fixture subsequently passed its 48
existing accepted cases and its new applied-moment, invalid-transverse-force
and restoration assertions. The exact output, log and input hashes are
`slider-owner-reaction-native-001.*`. This is still an owner-only result, not
a Matter contact or coupled rollback qualification.
