# Parent-refined skin/fat preparation

The existing preparation owner can now receive a validated parent-refined tetrahedral topology instead of re-extruding its refined surface. This permits local joint resolution changes while retaining original material regions, source-face identities, physical mass accounting, contact ownership and the existing Metal solver. The descriptor is optional; callers that omit it retain the canonical path.

Children must retain positive geometry, mapped parent containment and volume closure, material identity, exact registered-surface face rows, stable face identities, consistent shared-face winding, and complete outer/inner/sidewall boundaries. Existing Float32 emission, thickness, density, donor debit and contact checks remain active. No runtime force path or alternative simulator is added.

Validation on SSH Numi:

- Strict C++23 warning-as-error synthetic tests cover true edge bisection, missing volume, outside-parent children, invalid ancestry, boundary orientation, and projected energy/force/tangent consistency.
- The actual 2,332-row Human skin source gives exact default-versus-explicit-identity parity for all 6,996 coordinates, 27,714 tetrahedra, material rows, contact nodes, source ancestry and donor mass.
- Existing precomputed-layer and candidate-builder tests pass. The from-layers test contained a stale expected error substring; its rejection behavior fails identically on the old and new headers. The assertion now checks the actual face-winding rejection, retaining the rejected append and world-size checks, and passes against both headers.

This is preparation capability, not admission of a repaired Human or natural joint motion. The generated elbow variant is still undergoing actual source-registration and layer-owner checks. Its failed duplicate-face, surface-row and inconsistent outer-coordinate trials remain retained; none are substituted for the baseline. The separate ankle shape candidate runs for 1 ms but fails regional force/strain comparisons and remains unadmitted. No supported-rest, 16-second, 60-second, clinical or performance qualification is claimed.

Receipts record exact commands, source hashes and limitations. publication-inputs.json identifies the narrowly published inputs. The shared dirty checkout, normal Git index and working branch are preserved.
