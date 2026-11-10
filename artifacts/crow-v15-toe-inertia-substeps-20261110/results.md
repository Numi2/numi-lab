# Toe armature and timestep interaction results

Neither lower-armature candidate passed any of eight flight lanes at 16 physics substeps. Both candidates had at least one toe at its 12 rad/s speed limit in all 128 sampled lane/time points. The 5e-6 candidate had no physical or solver failures but missed flight tracking; the 5e-7 candidate had five physical failures. Neither is safe to promote. These tests do not establish realistic contact, grip, landing, or navigation.

The run log also records initial missing Python dependencies and a duplicate original/toe 16-substep screen caused by a script-edit error; those redundant outputs are kept, and are not counted as new candidate results. Dependencies for the tested host libraries and shaders are archived in the toe-inertia and toe-free-flight artifacts. Newly fingerprinted zero policies are included here.
