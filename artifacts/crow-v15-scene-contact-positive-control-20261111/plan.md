# Constructed scene-contact observer control

Place one training gate deliberately close to the bird at reset to force a bird-to-scene contact. This is an artificial observer positive control, never a navigation success or held-out layout. Replay identical starts with and without the new scene-contact observer, compare accepted q/v and existing contact summaries, and require a logged substep collider pair whose nonzero impulse agrees with the accepted endpoint contact witness. A solver failure or missing contact makes the control inconclusive.
