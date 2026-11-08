# Active-head navigation transfer and full-journey trial

The source was the unqualified 807-input, 15-action actor previously transferred
from the older retained v10 policy. The compiled task check confirms inherited
action ordering and current-observation operators are preserved by the v15
task. This diagnostic inserts nine zero-connected head observations immediately
before the 600 visual inputs, appends three neutral action outputs, and binds
the resulting policy to the new native contract. On 16 synthetic mapped inputs,
all 15 inherited raw output logits matched exactly and all new output logits
were zero. This is a network projection check, not evidence of physical or
visual behavioral equivalence: the articulated body and camera differ.

Two 8-lane exploratory comparisons, on layouts named held-out-a and held-out-b
with one seed each, used the full-journey difficulty band 10, 1,600 steps,
four physics substeps and no scheduled resets. The fresh actor, the two-update
smoke actor, and the transferred actor each reached zero waypoints; all first
episodes physically failed. The transferred actor lasted substantially longer,
ending in contact around steps 1359–1411, but is not successful navigation.
These layouts and seeds have now been inspected and are development evidence,
not untouched qualification data.

An exploratory fixed-learning-rate continuation used a fresh v15 critic and
fresh exploration head, 16 environments, 128 steps and 16 updates on the
training course at band 10: 32,768 samples, zero failed simulation steps. The
initial attempt to construct a stochastic pack failed because the legacy actor
was deterministic; the failed attempt is retained. The corrected continuation
still reached zero waypoints on both paired layouts, with all first episodes
ending in physical failures. No candidate was promoted.

The fixed promotion gate remains at least 28/32 first-episode five-waypoint
completions on each of three untouched randomized-course seeds, at least 90%
aggregate and zero failed simulation steps. The present trials are far below
that gate. The body parameters and camera are engineering estimates, and
successful navigation alone would not qualify other lifelong crow capabilities.
