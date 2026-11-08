# Residual teacher physics-failure boundary

The direct residual-teacher distillation path previously masked failed rows only when constructing the final loss weights. It computed per-environment mean rewards and global reward ranks before that mask, and also used per-environment aggregate evidence. Consequently the content of a rejected transition could affect the weights of accepted samples. The regular native search wrapper already rejects a reported failed environment step, and PPO already rejects rollout packs containing physics errors; the separate direct `distill` entry point did not enforce that boundary.

Distillation now rejects any rollout with a `physics_error` before reading evidence, ranking or creating the learner. It also rejects evidence reporting any nonzero per-environment physics-error count, even if the supplied rollout reports none. It retains the existing finite-value and dimension checks. This deliberately rejects the full mixed pack: aggregate evidence does not currently establish accepted-only provenance. Training healthy rows from such a pack requires a separately qualified accepted-only evidence and transaction path.

The regression uses a two-step, two-environment synthetic rollout and mocks the native pack reader and learner constructor. Before the fix, failures at either tested position reached the learner (`baseline.log`, from the earlier version of the regression). After the fix, both are rejected; contradictory failure evidence is rejected too; a clean rollout reaches the unchanged learner path. No output artifacts are created in these tests. Six residual-teacher tests, 11 existing HyperPolicy failure-boundary tests with MLX on the Apple M6 GPU, and one generic policy-evidence test passed. Logs, source hashes and test commands are retained here.

This demonstrates the Python admission boundary and its clean-path control, not a real native failed rollout, a completed training update, trained skill, accepted-only mixed-batch support, sensor transactionality, or force correctness. Both experimental collision/contact options remain disabled.

```sh
PYTHONPATH=python /Users/numi/numi-window-integration-venv-20261107/bin/python tests/solver_residual_teacher_test.py
PYTHONPATH=python NUMI_TEST_MLX_DEVICE=gpu /Users/numi/numi-window-integration-venv-20261107/bin/python tests/hyper_policy_failure_training_test.py
PYTHONPATH=python /Users/numi/numi-window-integration-venv-20261107/bin/python tests/policy_generic_evidence_test.py
```
