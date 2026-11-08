# American crow native episodic object practice

Source: `51c05ad3` on `agent/american-crow-rl-20261107`, macmini,
8 November 2026.

The optional `--object-push-curriculum FILE` adds repeated motor-learning
opportunities to ordinary v14 native rollout and recurrent MLX training. The
canonical configuration has160-tick episodes and independently samples object
x[.223,.227]m, y[.028,.042]m through existing generic RealityPack operators.
It preserves authored height/orientation/mass/material and zero launch velocity.
Episode reset restores body/scene/camera/GRU; model/Adam and accepted lifetime
ticks continue. The original fixed fixture keeps its100000-tick horizon.
Changed task fingerprints require fresh policy binding. Actor observations
remain proprioception plus genuine delayed RGB; object truth stays in the critic.
No per-step host loop, alternate simulator or new physics kernel is introduced.

## Engineering verification

- Full131-target Release build and four existing native regressions pass:
  life checkpoint, recurrent rollout, retinal API and temporal history.
- The33-lane native instrument crosses a SIMD boundary and produces99 distinct
  lane/episode placements over three32-tick episodes. All are within the
  canonical placement ranges. Neutral horizontal drift is below10um and
  settling-depth error below0.5mm. Real RGB varies with placement, and camera
  validity clears for exactly two ticks after reset. Lifetime clocks reach70.
-6930 native control transitions check chunk1/8 stream equality, final native
  components and two fresh35-tick continuations from tick35 across reset64.
  All six malformed configuration controls reject. The legacy fixed fixture
  rejects a curriculum-bound policy. No physics rejection or host reset occurs.
- Ordinary recurrent learning compares three97-tick updates with one plus two
  resumed at chunks1/8. Five shared boundaries match all41 model/Adam arrays and
  metadata, canonical policy/rollout bytes and native world/retina/GRU state.
  The final lifetime clock is291 ticks per lane. Six invalid restores reject.
- A345-tick scripted recurrent control matches186-tick futures from tick159
  at chunks1/8, including resets160/320, exact observations/latents/logprobs/
  values/transitions/memory/reset masks/outcomes/bootstrap and native components.
  GRU clears exactly at reset and the camera restarts for two ticks. Lane0's
  three sampled positions and first-episode0.265678299m maximum movement remain
  in the raw summary. This is scripted input, not learned manipulation.
- The original fixed-fixture recurrent admission passes without the new flag.
- Changing the curriculum y upper bound to.041m rejects the97-tick checkpoint
  with `Training checkpoint scheduler/learner configuration mismatch.` No new
  policy/rollout/checkpoint publishes. The initial analysis expected the words
  `configuration differs`; its source/failure are preserved. Reanalysis verifies
  original stdout/stderr receipt hashes and the actual diagnostic, with no rerun.

The new CTest wrapper is registered and compiled, but its separate CTest
invocation is not claimed: the same native executable runs through the official
Numi admission command with exact loaded-library verification.

## Bounded performance and recovery

One128-lane process accepts40960 samples over two160-tick updates, including the
native episode reset, at711.496 transitions/s in57.569s. Full coupled checkpoint
publication takes0.135634417s total. Each checkpoint directory is131421753B.
MLX peak is580314256B; native reported retained buffers104109362B and transient
private buffers76476188B. MLX counters do not measure whole-system peak. Swap is
zero before/after and zero native steps reject. Raw performance evidence is
380874954B before its summary, within the prospectively declared512MiB limit.
This short measurement does not establish sustained throughput or a controlled
learning-algorithm speedup.

`manifest.json` binds all selected raw evidence and retained runtime by path,
hash, mode and mtime. Content-deduplicated compressed chunks reuse previous Git
archives without rematerializing them. `runtime-recovery.json` identifies the
exact executables/library/three shaders, committed Python and canonical assets,
curriculum, prefix life/policy, coupled checkpoint and expected futures.
`restore_runtime.py` reads actual committed Git objects. `verify_recovery.py`
relocates both curriculum arguments and executes fresh fixed-policy and coupled
training continuations. It compares canonical rollout/policy bytes, all native
components, model/Adam arrays and metadata and scientific learner metrics.
Process-dependent timings and allocator counters are recorded separately.

Recovery results belong in `verification.json`. Raw deletion requires verified
committed objects, actual successful restored-runtime execution, remote HEAD
agreement, unchanged hashes/modes/mtimes and closed handles. Exact cleanup
receipts preserve the deletion sets. Archive checkout duplicates may then be
omitted while remaining recoverable from Git. Shared Git providers and unrelated
work are protected.

## Research scope

This completes an explicit episodic practice interface in one development
apparatus. The wider API guard x[.222,.228]m, y[.026,.060]m,32..4096 ticks is not
full physical qualification of that envelope. Existing32/64-substep sensitivity
remains. Learned visual manipulation, held-out layouts/materials, solid-food
pickup/carrying, unbroken ecological needs/resources and social learning remain
open. The scripted policy cannot qualify those outcomes.

The user's three optimizer questions are addressed separately in
`artifacts/crow-rl-optimizer-audit-20261108`: final per-factor KL after the final
optimizer step, retained-rollout head advantages versus longer returns, and
actual actor movement after shared actor/critic clipping. The audited recipe is
gamma0.999/lambda1 with full320-tick sequences. No adaptive learning-rate trial
is selected. Behavioral selection remains held-out cue-conditioned choice and
intake.
