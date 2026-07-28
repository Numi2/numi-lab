# Numerical contract

- GPU physics is FP32. Metal does not provide native shader `double`.
- Positions use metres, time uses seconds, mass uses kilograms, angles use
  radians, and forces use SI units.
- Quaternions use `(x, y, z, w)` and are normalized after composition.
- Joint integration is semi-implicit: velocity advances before position.
- Control-rate `dt` is divided into a fixed number of physics substeps.
- Non-finite state is terminal and the affected environment is reset.
- Actuator effort, joint velocity, and joint position are bounded by the
  compiled model.
- Contact uses bounded compliance in the first implementation. A warm-started
  complementarity constraint solver is the next contact milestone.

The CPU reference consumes the same FP32 model records but evaluates dynamics
intermediates in FP64. It constructs the reduced mass matrix through repeated
inverse dynamics and solves it with Cholesky. The Metal runtime instead uses
the linear-time articulated-body algorithm in FP32. Their one-control-step
comparison is a convention and equation check, not a bitwise-equivalence
promise.
