# Final-policy decision-group KL feedback: behavioral trial failed

The matched32-update development trial rejects the predeclared adaptive benefit
prediction. Initial/fixed/adaptive Gaussian policies score371/482/466 correct
head-side choices and69/105/98 positive-intake lives out of576 each, on9 unused
allocation/action seeds in the same fixed head-reachable two-cue fixture.
Adaptive versus fixed: choice−16 (−2.78pp), seed sign-flip p=.015625;
intake−7 (−1.22pp), p=.6328125. Choice decreases in7 seeds, ties in2, improves
in0. Intake increases in4/9. Both co-primary gain criteria fail. Intake remains
inconclusive; this does not establish equivalent intake or universal optimizer
harm. No policy promotion, retry, budget extension or further heldout tuning.

Both arms start from the identical archived stochastic policy65, critic,
normalization and exploration, with fresh Adam and matched training/learner
seeds. Only adaptive feedback differs. Both process655360 accepted training
transitions in171.1s (~3830/s), zero physics failures. Combined evaluation is
2211840 transitions. All native causal/intervention checks pass; intact cues
favor adaptive choice/intake over GRU-clear, absent and swapped controls.
These are evaluation-seed results conditional on one matched training pair,
not independent training replicates or broad algorithm/ecology qualification.

## The three optimizer checks

Final KL is recomputed after both full-batch Adam steps, separately for head15
and beak17, with64 and1024 actual selections. The complete native selection
mask separates64 rare head-plus-beak decisions from960 beak-only decisions.
KL is old-to-final latent Gaussian KL before physical action mapping/clamping;
it is a diagnostic, not intake success. The opt-in controller holds one rate
through the whole update and changes only the next rate from the maximum
conditional joint group mean. Existing rate bounds/rule remain; no hard KL
constraint. Legacy/default optimizer behavior is unchanged and the retained
65→66 update reproduces all53 model/Adam tensors and configuration bytes.

Actual gamma=.999, lambda=1; a128-tick coefficient is.879797, rather than
.000388979 for gamma=.99/lambda=.95. Final head advantages correlate with
reward-only Monte Carlo at.99883 fixed/.98406 adaptive. Their means for
intake/no-intake are+2.8457/−.6031 and+2.3201/−.5744 respectively. Independent
FP64 lambda1 return identities agree within1.84e−5/2.26e−5. Useful reward
credit exists here; full-sequence backpropagation is not used as an attribution
argument. Earlier three retained audit cases remain in the optimizer-audit archive.

Actual final whole-update actor Adam displacement is.06222 fixed versus.39407
adaptive. Final head KL=.00001975/.00147094; beak KL=.00036167/.00808796.
Fixed final steps have no clipping. Adaptive final critic norms2.365/2.266
versus actor.102/.126 cause joint scales.4224/.4407; clipped actor norms
are.0430/.0555, with applied Adam displacements.20188/.19445 per step.
The full histories retain actor/critic norms, clipping and actual displacement.
Rate amplification produces larger updates but fails the behavioral criterion.

## Verification and recovery scope

Full Release and21 focused Python tests pass. Failed helper/test attempts are
retained: import typo, unbound toy policy identity/semantic contract, missing
restore parent and mistaken numbered final checkpoint. Final files live at
each arm root; interval31 writes checkpoint96 only. The corrected offline
last-update replay reproduces both53-array model/Adam states and configuration.
A separately sealed engineering continuation from each native checkpoint96
reproduces all4 canonical final policy/deployment/rollout/learner files exactly,
with40960 replayed transitions and zero physics failures. It also emits full
final native checkpoints97 with explicit interval1. Process-local incumbent
selection is excluded from that continuation comparison. These replays do not
add independent trial observations or change the frozen primary endpoints.

`manifest.json` recovers every raw plan, command, outcome, failure, checkpoint
and exact runtime through content-deduplicated Git objects. `runtime-recovery.json`
binds native binaries/shaders, actual Python source blobs and product inputs.
The executed Git-restored check independently replays both actual final learner
updates and one complete adaptive12-arm causal block (3 of9 evaluation seeds),
with95 exact behavior/binary files and12 path-bound visual manifests checked
by native owning identity functions. See verification.json for the final receipt.
The complete32-update histories and all9-seed evaluations are not re-executed
by this scoped recovery. Raw/duplicate cleanup requires verified actual Git
bytes, unchanged source hash/mode/mtime, no open handles and remote HEAD parity.

Protected deterministic feeding deployment97 is unchanged. Reliable stochastic
feeding, new geometry, solid-food handling, navigation/landing, social learning,
physiology/weather and the full lifelong embodied goal remain unfinished.
