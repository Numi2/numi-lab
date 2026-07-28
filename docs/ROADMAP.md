# Heavy-lifting roadmap

## M0 — Franka reach on Metal (working vertical slice)

- Fixed-base articulated-body dynamics with velocity bias and gravity
- Joint position/torque drives, effort and limit handling
- Primitive ground contacts and device-resident resets
- Batched Franka reach observations, rewards and MLX PPO
- Native throughput, CPU/Metal parity, and M4 validation record

## M1 — Unitree G1

- Floating-root pose, twist, inertia, and integration alongside the existing
  29-joint tree capacity
- Pinned G1 model, actuator semantics, and collision-filter provenance
- Broadphase pair generation and sphere/capsule/box/convex narrowphase
- Warm-started frictional contact and equality/joint constraints
- Self-collision filtering, foot contacts, IMU and projected gravity
- Flat-ground velocity tracking, then rough terrain
- Domain randomization and policy export for simulator-to-simulator checks

## M2 — Franka manipulation and model import

- Free 6-DoF object bodies and object-to-robot contact
- Franka push, lift, and place tasks with contact/force sensors
- URDF and MJCF import into the existing immutable model format
- Pinned MuJoCo and Genesis trajectory/contact comparisons

## M3 — Simulator platform

- Metal raster viewer with contact, constraint and reward-term inspection
- GPU depth, segmentation and ray/LiDAR sensors
- Manager-style action, observation, reward, event and curriculum modules
- Replay, rewind, state serialization and deterministic diagnostic mode

## M4 — Broader physics

- Heightfields and triangle meshes
- Tendons, cables and deformable/particle solvers behind solver interfaces
- OpenUSD ingestion and tiled camera rendering
- Multi-Mac rollout workers over the MLX distributed layer
