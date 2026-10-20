# Near-tangent supported trial
Change only the common initial root Z from 0.10571993142366409 m to
0.10583633184432983 m (10 micrometre initial support clearance), plus the
28-row catalog and bitwise-checked parallel tangent cache. Orientation,
joint configuration, zero initial velocities, gravity, contact settings,
material properties, tissue mass and ownership remain fixed.

Run fresh initial states for 8 and, if admitted, 32 steps at 125 microseconds.
Stop on first physical rejection. Require finite state, positive element
Jacobians, mass accounting and zero root assistance. The 32-step trial must
show real bed contact impulses to establish contact engagement. These
1/4 ms development trials cannot establish sustained support, breathing,
physiological plausibility or whole-body tissue-contact completeness.
No visual checks. Preserve failures, do not relax solver tolerances.
