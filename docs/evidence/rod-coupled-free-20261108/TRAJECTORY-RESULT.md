# Free-end 3D trajectories — 8 November 2026

All three preregistered 10-step gravity trajectories completed and passed the original reference and local-balance gates, with byte-identical results across two runs. The FP64 reference and Metal API advanced from their own preceding states. These are 30 accepted steps per run, not 30 independent trajectories.

The largest position difference was 9.07130798e-07 m against the 1e-6 m limit, in seed 92101 at step 10. The largest velocity difference was 0.000201642446 m/s; twist and twist-rate differences were 2.50972293e-07 rad and 5.82103811e-05 rad/s. Independently recomputed local linear and angular residuals were at most 0.0002011917 m/s and 0.000226577865 rad/s. Maximum one-step and accumulated momentum-balance errors were 3.00689266e-13 and 3.91817535e-13 kg m/s respectively. All local perturbation negative controls were detected. Input, output, binary and plan hashes and both runs are retained.

The position error uses approximately 91% of the declared limit, so longer trajectories and error accumulation need explicit study. This is still a homogeneous research-only rod without contact or attachments. It does not exercise MetalWorld checkpointing, rollback, coupled contact, tissue, or calibrated surgical physics. The initial unglobalized FP64 reference failure is preserved separately.
