# Sensor step transaction: owner and publication design

Status: design only. The tracker does not yet provide rejected-physics replay.
The existing encode-cadence fix covers an encoder rejection, not a subsequent
command-buffer failure or a GPU-reported physical failure.

## Current ownership

MetalWorld encodes task observation, projects the current body poses, then
invokes MetalHybridObjectTracker before action selection and physics. The
tracker currently changes its object-track and depth-ring buffers on the GPU
and changes a single host observation/sensor sequence while the batch is
encoded. The renderer retains depth, validity, identities, camera state and,
for the fast path, mesh winners. Those are reused when host cadence says a
frame is held. A later physical status can differ per environment; task
completion sees it only after physics. On a physics error, task completion
marks the episode terminal and sets pending reset while the world reports
failure. Thus an observation after that error must either be withheld, or
start a new episode from the reset state; it must not silently continue using
the failed step's sensor history. Exact retry of the rejected state requires
a separate explicit owner-level continuation contract.

There is an additional timing hazard: renderer frameSeed mixes both the
global frame index and global sensor sequence. Simply moving the history ring
behind a per-environment success check still changes future noise after a
mixed-batch failure. The observation corruptor also uses episode identity and
step. Reset, dropout and source timestamps must be treated as state, not as
metadata reconstructed from a global attempted-step count.

## Candidate transaction

For each environment, the tracker owns one accepted sensor state and one
scratch candidate. The accepted record includes:

- sampled frame data needed by both paths: depth, validity, identity where
  object tracking or the oracle needs it, camera pose, and direct-winner data
  or an equivalent materialized metric-depth plane;
- object-track records, the complete masked/raw depth ring and their current
  episode identity;
- accepted physical-step clock, exposure sequence, capture time, held-frame
  age, and the full noise seed/counter inputs.

A pre-observation pass initializes candidate from accepted state. It observes
the current pre-step geometry, acquires a new frame only when that
environment's accepted clock requires one, otherwise reads its accepted held
frame, and writes only candidate history and the temporary policy observation.
Per-environment counters and episode identity drive renderer and observation
noise. Frame geometry can be generated in a batched pass but sensor sampling,
held-frame selection and noise keys must come from the individual candidate;
the current uniform-only global frame index is insufficient.

After physics and before the next control-step observation, a GPU owner hook
reads that environment's actual world and contact status. On success it
publishes the complete candidate record as the accepted record. On failure it
discards the candidate and publishes no observation as accepted. These are
one logical operation: copying only histories while committing cadence or a
held frame is invalid. Any buffers returned to policy training are marked
provisional until the corresponding step succeeds. If physical failure marks
the task terminal, the failed record is withheld and the next observation is
permitted only after a successful reset has started a new episode. Reset
stages a fresh sensor record and fills its history without borrowing a held
image or track from the previous episode; a failed reset must not publish it.

The status-dependent hook belongs in MetalWorld's per-control-step command
graph, after the same status writers used by task completion and before the
next observation. The callback contract must cover both depth and object
tracks; it must not special-case the new depth mode. The tracker must own
buffer sizes, allocation failure, initialization, and generation identity.
Command-buffer failure and abandoned asynchronous submissions require an
owner-level transaction too: no candidate or host cache may be presented as
accepted when completion is unknown. MetalWorld currently reports errors at
submission wait; the design must account for abandoned submissions and
multiple queued contexts, rather than relying only on a synchronous caller.

## Required evidence before claiming repair

1. In a single environment, accept two steps, force a failed physical step
   after observation, restore/retry from the same accepted state, and compare
   all sensor state and output bitwise against a run without the failed
   attempt. Include a slow sensor held across the failed step.
2. In one multi-step, multi-environment batch, force a failure in one
   environment only. Verify the other environment publishes every accepted
   state and that the failed environment's sensor record and noise sequence
   match the declared termination/reset or retry contract. Test both direct
   winner and normal renderer paths.
3. Cover raw depth, oracle depth and object tracks, reset on and adjacent to
   failure, full and pixel dropout, held-frame age, episode identity,
   timestamps and reproducible noise. Inspect source state and all retained
   buffers, not just the visible actor tensor.
4. Inject observation-encoding failure, command-buffer failure and abandoned
   asynchronous submission. Verify failure reporting and that no later
   submission can reuse uncommitted state.
5. Compare single-step against multi-step submissions with the same accepted
   steps and seeds, then perform complete MetalWorld rollout and restore
   qualification. A synthetic kernel test by itself does not satisfy this.

The earliest bounded implementation should be gated until these tests pass.
Neither the existing host counter nor the isolated raw-depth kernel test
proves the candidate transaction.
