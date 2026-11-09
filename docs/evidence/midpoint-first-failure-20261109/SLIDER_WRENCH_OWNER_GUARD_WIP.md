# Slider moment owner compatibility: staged, not qualified

The staged Matter adapter advertises a body wrench for the slider contact
moment, but the MetalWorld capability validator previously rejected any
midpoint owner that advertised body wrenches. This made the proposed coupled
program invalid before execution. The restricted owner capability now permits
that combination. Its GPU pre-commit status promotion checks that all applied
forces and the fixed-root torque are zero and the moving body's torque is
finite, so the generalized axial effort is not also applied as a body force.
The newly named guard kernel prevents an older Metal library from silently
omitting that check. The separate Matter runtime research guard remains closed.

The CPU adapter/ownership guard passed on the Apple host; executable SHA-256
`408390e613d9031189f4f8b6c8aa020675c87b10f0af687cc32313ab1cb8d305`.
The host and complete MetalWorldContact shader translation unit passed syntax
checks. A nine-case native GPU guard test passed using an isolated compilation
of the exact kernel extracted from that translation unit. It covers zero and
finite moment, extraneous forces and root moment, nonfinite values, disabled
wrench input, and preservation of prior failures. Executable SHA-256:
`839fa874faec70f5a6faeaa2cddbbec026da4bc107195367f69bb01c043964f4`;
extracted source SHA-256:
`6abb67807860270b04b218a9f480955551c9b0d20cbdf0e3199b851621ba02ea`;
compiled isolated library SHA-256:
`545e098eb707bf6fe5d160e0d2f26ca7f8fed595629da3299a59cba66b45c1ee`.
The full MetalWorld library and coupled path have not been run with this change.
No support-reaction measurement or coupled transaction is established.
