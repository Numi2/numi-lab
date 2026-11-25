# Matched free-fall cross-engine correctness gate

Question: do Numi MetalWorld free-motion ABA and pinned MuJoCo 4.0.0 reproduce the same discrete ballistic update before any timing comparison?

Both cases use one 1 kg floating body, zero initial velocity and zero damping, initial y = 10 m, gravity (0, -9.81, 0) m/s², 1,000 steps of 0.001 s, no applied force, no collision, no rendering or controller. Numi uses one environment, one physics substep and free-motion ABA. MuJoCo uses a free joint with Euler integration and the same physical body inertia; contact is disabled. Rotations remain identity. FP32 Numi and FP64 MuJoCo are reported separately.

The independent discrete semi-implicit Euler reference is `v_y(n) = -9.81*n*h` and `y(n) = 10 - 9.81*h*h*n*(n+1)/2`. Before inspecting results, require each engine's final position and velocity to be within 0.001 m and 0.001 m/s of this reference, and no rejected Numi steps. Report actual values, runtime/source/binary hashes and any failure. No timings or superiority claims follow from this gate; matching a no-contact trajectory does not establish contact, articulated, force, learning or sustained-performance equivalence.
