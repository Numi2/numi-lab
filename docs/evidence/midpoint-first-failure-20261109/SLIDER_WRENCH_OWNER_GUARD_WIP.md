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
checks. A nine-case native GPU guard test has been built (SHA-256
`839fa874faec70f5a6faeaa2cddbbec026da4bc107195367f69bb01c043964f4`)
but not run; it requires a newly built MetalWorld library. No support reaction
measurement or coupled transaction is established by these checks.
