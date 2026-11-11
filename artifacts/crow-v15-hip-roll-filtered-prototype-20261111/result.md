# Hip-roll prototype with preserved anatomical collision exclusions

This revision restores only the two original torso-to-thigh collider exclusions lost when inserting proxy joints. collision-filter.txt lists exactly the torso/left-thigh and torso/right-thigh shape pairs. All other collision pairs remain as before. The revised prototype passed the same 16 neutral-pose kinematic comparisons and 12-case CPU/GPU free-motion comparison as the failed-filter predecessor; the worst scaled acceleration difference was 7.80e-8. It has two added physical joints, two proxy bodies totaling 20 mg, and is not dynamically identical to the 33-body original.

The companion contact test runs beyond the initial interval, unlike the predecessor, but support, release, toe speed, flight, walking and navigation require their own evidence. The free-motion result alone does not qualify this model or its controller.
