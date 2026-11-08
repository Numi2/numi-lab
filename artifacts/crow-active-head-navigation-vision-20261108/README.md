# Active-head navigation: visual binding repair

The legacy course packs bound gate/slalom/perch instances to global body
indices 15–19. The active-head embodiment inserts three articulated bodies:
those indices include the head and lower beak. Consequently, two gate visuals
were attached to the bird and obstructed the head camera. The retained
obstruction controls precede this finding.

The corrected packs preserve legacy geometry and bind the five course bodies
to indices 18–22. The native runtime validates these indices against current
body names and rejects stale course packs. The generator rejects unexpected
legacy bindings. The policy observation fingerprint now binds the v15
policy-visible camera and visual assets, while excluding capture dimensions,
capture enablement, and retention budget. Earlier v15 diagnostic policies are
incompatible by design. Older task contracts are unchanged.

Four-step paired zero-action native controls completed with finite depth and
no failed transition. The corrected camera frame shows course posts, and
near-depth counts decrease from 132/144 in the obstructed prior view to 13/144
on the corrected first frame. The no-robot and shifted-camera controls are
retained. Rejection tests verify stale body bindings and shifted cameras fail;
the same sensor without recording still loads. The unit compile check verifies
camera and asset changes alter policy identity and capture-only changes do not.

This is a bounded visual plumbing check, not anatomical eye calibration or
proof that vision supports navigation. The raw captures and executable outputs
are archived separately in the Git evidence payload.
