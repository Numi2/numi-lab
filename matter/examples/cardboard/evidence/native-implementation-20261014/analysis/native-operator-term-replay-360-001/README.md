# Native operator term replay

This diagnostic uses the exact source101 production Metal element operator and
nodal gather with the sealed capture360 payload from package003. Cached and
fresh normal paths are unchanged; raw four-node element directional forces are
also exported. A third call zeroes only its private element-output buffer before
the same gather, isolating mass. No constitutive or accepted history is changed.
No Matter transaction is executed. The mass-only control is an operator-term
diagnostic, not a physical model. Normal combined actions must exactly reproduce
native368 before interpreting the terms. Force sign is unchanged:
A(delta_v) = M(delta_v) - dt^2 * d(force)/d(delta_v).

Element arrays have little-endian float32 shape(2,1920,4,4), four node slots
with xyz forces and w pressure. Nodal arrays have shape(2,882,4). The native
binary accepts compiled-package, prepared-payload, output-directory; optional
--preflight-only performs immutable input checks without creating a Metal device.
Compilation and CPU preflight do not dispatch the diagnostic.
