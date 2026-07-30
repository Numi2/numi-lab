# MetalRobo tactile Swift bridge

Add `include/metalrobo` to the target's C header search paths so Swift can
import `MetalRoboC`, link `libmetalrobo.dylib`, and add
`MetalRoboTactile.swift` to the application target.

The wrapper borrows `MTLBuffer` and `MTLComputeCommandEncoder` objects. Its
normal `encode` path does not commit, wait, allocate, or read back; the caller
owns command-stream ordering with physics and the policy graph.

Construction requires the same explicit authored `.mrworld` pack consumed by
MLX. There is no built-in Franka scene, collision-derived adapter, or fallback
pack path.
