# Native retinal stream: timing admission

The lifelong American-crow project now has a native RGB-only timing layer,
not a complete visual learner. The existing v12 actor remains unchanged at
97 non-image inputs. The renderer and this stream have not yet been composed
inside accepted rollout/inference. Geometry and head-camera calibration are
in the separate crow-rl-head-camera-20261107 archive.

RetinalStreamContext owns private GPU pending/visible frames and timing state.
It borrows RGB, native task episode/initialization fields, actor output and the
caller's command buffer. It encodes area averaging, fixed c/(1+c) radiance
response, rational sampling, delayed delivery and publication. No commits,
waits, host reads or host sampling decisions occur in encode(). No depth,
instance identities, segmentation, poses or object coordinates enter it.
Setup and explicit publication may synchronize; neither is a per-step loop.

The development schedule is 3/10 samples per 50 Hz accepted tick (15 Hz),
two-tick delivery latency (40 ms), two eyes and explicit valid/age features.
Repeated ticks are idempotent despite changed RGB. Episode changes clear
pending and visible images for the affected lane. Clock gaps, backwards clocks,
uninitialized clocks and invalid captured radiance latch native per-lane errors;
publication becomes invalid/zero. The tested negatives are clock gaps,
nonfinite RGB, and finite-radiance reduction overflow. No reset clears errors.
One command buffer at a time may mutate a stream; failed GPU commands poison
the context. A debug snapshot is not a checkpoint and cannot restore images,
world, policy or memory. All rollout acceptance must inspect stream errors.

Trial 01 passed the original instrument. Trial 02 prospectively extended
parameter boundaries and finite-radiance overflow controls. Trial 03 added
source-image index admission and raw output retention before assertions.
All retained trials passed their preregistered predictions. Current controls
cover three GPU-fixture lanes, fifty ticks with a duplicated seventh tick,
a single-lane episode reset, exact chunks 1/8, and an independent FP64 analytic
area/timing oracle. Maximum feature error is 4.77e-8 against the registered
3e-6 limit. Ratios 1/1 and 1/50 and delay boundaries 0, 1 and 16 ticks pass.
Eight malformed configurations reject, including an overflowing source-index
configuration, as do short/null output, null clock and missing-eye calls.
Main, boundary and negative raw actor planes, guard features, state and error
records are retained. Output layout metadata is authoritative for trial 03;
earlier layout files predate per-file call counts. These are native synthetic
instrument controls, not embodied trajectories or independent animals.

All eight focused native CTests and 19 Python crow checks passed. Their retained
logs and copied CTest evidence remain bound to this source. The earlier full
suite before the source-index guard is retained separately too. The frozen
feeding actor still exactly matches its earlier ordinary/residual replay with
the initial deterministic base. The new sensory kernels are in a separate
RetinalStream.metallib; the established physics/policy metallib hash is unchanged.

Every outer trial retains an exact tracked patch and untracked source inputs
against Git commit 20327805; archive creation rechecked these hashes and native
stdout/stderr receipts. runtime-receipt.json identifies exact current binaries
and both shader libraries retained on macmini outside Git. Source and raw
controlled evidence are in Git. No learned visual behavior, biological eye
calibration, full-life checkpoint, ecological performance or qualification is
claimed. Next: bind a distinct actor sensor contract, compose both head cameras
and this stream on the native accepted timeline, verify causal observations and
occlusion, then measure full rollout and learning throughput before training.
