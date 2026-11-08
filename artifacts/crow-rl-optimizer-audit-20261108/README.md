# Crow optimizer audit: final-step KL, returns and clipping

The retained-data audit compares three exact one-update replay pairs (65→66,
81→82,97→98). All53 model/Adam tensors match the archived outputs byte for byte
in each replay. No diagnostic candidate was deployed and no learning-rate trial
was added. The data and source identities are in runtime-recovery.json and the
content-deduplicated Git manifest.

## Final policy movement

The legacy `kl_divergence` is an average of **pre-step** metrics. It is retained
for compatibility and the existing adaptive controller is unchanged. The new
`pre_step_kl_divergence` names this explicitly. `post_update_kl_divergence` and
`post_update_action_N_{kl,approximate_kl,clip_fraction,decision_count}` are measured
after every optimizer step has finished, over the fixed generating distribution
and the same recurrent windows. Action15 is head yaw, action17 is beak opening.
These are old||new **latent Gaussian** KLs, before affine action mapping/clamping;
counts include only that factor's real selected decisions, never held proposals,
padding, terminal previews or neutral ticks. For this fixture each life has one
head selection and16 beak selections. This is not a hard post-step KL bound.

| Generating revision | Legacy mean pre-step KL | Final pooled KL | Final head KL | Final beak KL |
|---|---:|---:|---:|---:|
|65|0.0000317835|0.000351980|0.000452833|0.000323677|
|81|0.00235143|0.0187541|0.0139266|0.0178837|
|97|0.00955927|0.0253365|0.0597409|0.0216027|

Thus the two full-batch steps miss substantial final movement in the old metric.
At97, pooled final KL also hides a larger rare-head change:44.53% of sampled head
factors exceed the PPO ratio clip interval, versus21.44% of beak factors.

## Reward attribution at the head decision

All three actual checkpoint configurations use **gamma=.999, lambda=1**, with
320tick sequences. The128tick coefficient is0.879797, not the defaults'0.000388979.
Recorded first reward delays after tick64 are129..253ticks across these batches.
Independent FP64 discounted reward-plus-terminal-bootstrap-minus-value identities
match the native-data GAE to1.9e-5,2.8e-5,3.7e-5 maximum absolute error.

Mean head advantages for intake versus no-intake lives are:
65: **+2.354 / -0.373**;81: **+3.159 / -0.816**;97: **+4.093 / -0.891**.
Actual head advantages correlate0.997..0.999 with discounted Monte Carlo estimates
that omit the terminal bootstrap. Recomputing default GAE on the same records
largely erases intake-related separation; it was not the configuration used here.
Full-sequence BPTT does not replace this return calculation.

Temporal attenuation is not the main explanation on these audited batches.
There remains noisy cross-action credit: at97,115/128 sampled head actions choose
the correct side, but only32 feed. All83 correct-side/no-intake lives receive a
negative head advantage, like the13 wrong-side lives. Correct side is a coarse
metric: angle precision and later jaw actions can still fail. This is not proof
that every no-intake life failed only because of the beak, or a reason to inject
hidden allocation labels into learning. Allocation labels appear in offline
evaluation only. The critic's head-time observations/values are identical across
lanes in this fixed delayed-cue fixture.

## Combined actor–critic clipping and actual Adam updates

The learner now retains raw actor/critic/joint norms, actual joint clip scale,
minimum scale, clipped actor norm, per-step actor parameter delta and total actor
parameter delta. Adam's moments mean its update is not simply LR×clipped gradient.

At65 and81, the joint clip scale is1 throughout: **critic clipping cannot explain
those updates**. Their total actor parameter displacements are0.09420 and0.54453.
At97, mean raw actor/critic norms are0.367/7.352; mean clip scale0.244 and minimum
0.0573. Mean clipped actor norm is0.08874; total actor displacement0.44087.

An offline intervention starts from the same model/Adam state, uses the same
retained batch, minibatch order and exact original per-step LR schedule, and
removes only critic contribution to actor clipping. The64-lane recipes remain
byte-identical. At97, actor displacement rises2.206×, with update-vector cosine
0.440 to the original; final head/beak KL becomes0.01820/0.01132 instead of
0.05974/0.02160. Thus critic scaling changes the update, including its direction;
raw gradient suppression alone does not predict policy movement or better feeding.
This diagnostic branch is not a selected/deployed policy or behavioral improvement.

## Matched held-out behavior, not KL-target selection

After the diagnosis, the frozen Gaussian policies were evaluated on the same
three previously unused seeds2651213602..604, with192 lifetimes per arm and four
native arms (intact,GRU-clear40,absent rendered cue,swapped rendered cue).
737280 accepted transitions; no optimization or retry. All engineering controls
passed, with zero reported physics rejection/termination.

| Revision | Intact correct head side | Intact intake lives | Food micrograms |
|---|---:|---:|---:|
|65|128/192|26/192|890000|
|81|158/192|33/192|1310000|
|97|169/192|37/192|1370000|

Both cue-conditioned correct-side counts improve from65 to97 (left54→80/94,
right74→89/98). Paired correct-side gain41 has exact discordant-pair p=6.92e-7.
The intake gain11 has p=.169;81→97 intake gain4 has p=.678. These stochastic
feeding gains are **not conclusive**, and no adaptive-rate success is claimed.
Training histories, durations and populations also differ, so this retrospective
comparison does not isolate a causal effect of adaptive LR.

The separately registered deterministic97 candidate's earlier192/192 fixed-fixture
intake result remains scoped to its two repeated cue conditions. It does not turn
this Gaussian comparison into a success or establish broader ecology. The remaining
lifelong environment, solid-object manipulation and social learning are unfinished.

## Verification and retention

Independent NumPy final-Gaussian KL/selected denominators/actual Adam displacement
checks and the recurrent/held-factor suites pass (17 tests); the full Release
build passes. Git recovery must rerun all three offline exact updates, clipping
counterfactuals, return identities and all three complete native evaluations.
Behavior files must be byte-exact; path-bound visual identities are recomputed by
the native owner, all other JSON fields equal. Temporary and original raw copies
are removed only after exact hash/mode/mtime/closed-handle and remote-Git checks.
The interrupted object-manipulation mechanism probes and authoring failures are
preserved in this archive as unrelated unfinished work, not as a shipped task.
