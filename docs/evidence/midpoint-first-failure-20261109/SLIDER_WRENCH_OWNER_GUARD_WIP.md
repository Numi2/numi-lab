# Slider moment owner compatibility: staged, not qualified

Historical first staging (superseded by the force-carrying staging below):

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

## Force-carrying staging

A further review found that applying only a synthetic generalized effort did
not expose the physical slider-side contact impulse to the owning body. The
staged adapter now transfers the reduced full-step normal impulse as a body
force as well as the contact moment. Before ABA, MetalWorld subtracts that same
axial force from the synthetic generalized effort; the owner therefore sees the
contact wrench once. The restricted guard permits a world-z body force and
rejects transverse force, root wrenches and nonfinite inputs. This supersedes
the earlier zero-body-force design; the earlier evidence remains above.

The full Matter and MetalWorld shader translation units and MetalWorld host
source passed syntax checks. On the Apple GPU, seven preparation cases and ten
status-guard cases passed using isolated compilations of the exact extracted
kernels, including positive/negative force offsets, invalid forces and prior
failures. The extracted source SHA-256 is
`98799292f72cf51b1fa5f72b61d710fe95d45cd2cfdc0d96a69d1fe0acedfbea`
and the isolated library SHA-256 is
`a91053c007c248f0abd0d430c6955bc066d1ac8c255b9609ed9275ead3d8e083`.
Preparation and guard executable hashes are respectively
`2fe53f70bee4e19815f3c4b0c7e5dd090033fbc1d7d7e467e5846dfe35e34b2a` and
`0ced40160370f45a67e06bd8b5c54dd7be380b2547ad3fc71a4cdacd47b98ba2`.

These are component checks. The runtime guard remains closed: no native
Matter/MetalWorld contact step, independent constraint-reaction measurement,
conservation qualification or whole-environment replay has been established.
