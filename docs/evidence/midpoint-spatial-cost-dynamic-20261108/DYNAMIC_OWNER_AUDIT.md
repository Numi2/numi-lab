# Midpoint dynamic-owner boundary

This is an implementation audit, not a dynamically coupled result. The
experimental midpoint option remains disabled and rejects coupled candidates.
An attempted articulated-plane fixture using the existing physically scaled residual policy failed world validation: that policy currently requires pure FEM mechanical rows and excludes nonzero rigid generalized capacity. The failed native guard attempt is retained in `dynamic-guard-attempt-001.log`. A genuine coupled extension needs an appropriate mixed-owner residual certificate, not a disabled accuracy check.

The existing reference-area IPC path requires a fixed, bodyless plane; its
midpoint specialization also rejects prepared-state and accepted-extension
owners. The ordinary FEM/articulated foot-pad probe is a genuine two-way owner,
but it uses the existing integration path and does not certify midpoint.

The current free-rigid candidate projects positions and orientations with
`bridge.time.y` (the physical step), whereas the FEM midpoint stage dispatch
uses half that step. Merely allowing a dynamic proxy through preflight would
therefore mix evaluation times. A valid implementation must instead give both
owners the same midpoint primal state, use the unchanged physical contact law,
and publish full-step impulses consistently from that state. The compensated reference-plane contact evaluator is also explicitly limited to a fixed, bodyless plane; a dynamic candidate would use the generic gap path. A dynamic implementation must independently check the same physical law and precision at its moving witnesses rather than assuming those paths agree. The rigid-body
orientation, inertia and angular momentum update must also agree with the
actual MetalWorld integrator and its gyroscopic terms.

For contact samples with full-step impulse J_i on FEM and witness points p_i,
a two-owner acceptance test must independently establish:

* sum of FEM and rigid contact impulses equals zero;
* sum of angular impulses about the same fixed origin equals zero, using the
  actual body witness and lever arm, including the case of an offset load;
* the separately measured interface work equals the sum of J_i dot the FEM
  midpoint velocity and minus J_i dot the rigid witness midpoint velocity.
  This relative work is not assumed to vanish for compliant contact;
* the dynamically evolved body and FEM satisfy their own complete discrete
  momentum equations and the gap requirement along the committed motion;
* a forced failure restores both owners and their accepted histories before
  retry, while failure category and subsequent physical state are independent
  of the selected offending diagnostic index.

A rotating plane cannot be certified by checking only the old and new FEM
node distances to a stationary plane. The committed relative motion needs a
conservative swept clearance certificate, or an explicitly narrower derived
motion domain with a proof. Whole-environment acceptance must occur only after
both owners pass it. The existing fixed-plane endpoint certificate does not
provide that proof. No dynamic feature flag was loosened by this audit.
EOF'