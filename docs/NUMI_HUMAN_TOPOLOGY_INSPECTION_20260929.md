# Numi Human native anatomy inspection selector

The visual probe accepts repeatable `--hidden-anatomy-stable-id N` options when
`--torso-anatomy-payload` is supplied. It validates IDs against that payload,
rejects duplicates, sorts the selection, records it in the anatomy pose
snapshot, and clears only the instance visibility flag. Every source vertex,
triangle, primitive, and instance remains in the hashed native visual packet.

The topology-candidate ABI 5 payload has 598 surfaces: 579 unchanged source
surfaces plus 19 derived copies. The 29 September inspection selection hides
11 raw parents with exact self-intersection-free copies and eight derived
copies that failed exact self-intersection audit. The selection is stored in
the Human evidence receipt; reproductions must pass every listed ID. Rendering
an unselected profile can draw coincident originals and copies together.

This is a diagnostic visual selector, not anatomical or physical admission.
The old 579-surface ABI 5 packet and four rendered PNG views were reproduced
byte-identically with no selector. No physics runtime, shader, material, or
body binding changed for this feature.
