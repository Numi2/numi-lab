# Existing ordered limit solve with three equality rows per lane

All execution ran on the SSH Mac mini, Apple M4 Pro. Source023 extends the
existing SIMD limit solver from two to three equality rows per lane, covering
the resting 91-row block. No new buffer, physical law, impulse order, CPU
physics or readback is introduced. Larger blocks retain the existing fallback.

The paired six-second native runs 043/046 have bit-for-bit identical physiology
and surface traces, initial geometry pack, accepted packs at steps 639, 2783
and 2999, and terminal q/v. Rejected-step/retry checks pass and input hashes
remain unchanged. The already retained complete exact skin checks therefore
apply to those same geometry bytes; no duplicated geometry audit is inferred
for unrecorded states.

Measured complete-run GPU time falls 22.34%. Run046 takes 69.164 wall seconds,
or 0.08675 times real time. The 32-step profile places the finish stage at
3.77/4.11 ms, versus 8.44/8.08 ms before this change. The 51-row native32-step
regression is also bit-for-bit identical. Concurrent independent CPU anatomy
preparation was present; these are local measurements, not a hardware-wide
benchmark.

The manifest retains exact commands, source, binary, input and recording
identities. Full geometry and unretimed movies remain at the listed Mini
paths. This is the old thoracic anatomy with known lung/rib and cardiac wall
defects. It is not final anatomical, five-minute or real-time acceptance.
