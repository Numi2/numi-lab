# Integrated training-boundary verification

The reconciliation merge 53218a7ff1a30d914d71401aadb0a35a70bb98e4 combines main 8872881be with the existing product branch. On ssh numi, the same 11 targeted tests passed with MLX CPU and Apple M6 GPU; the generic policy-selection evidence test also passed. The retained main evidence checksums passed. These tests use synthetic rollouts and do not establish a complete GUI, native-training session, or policy skill.

The tested integrated sources have SHA-256:

- mlx_policy_learning.py: c37733101b9994635104291ddda68d648768b0dcefa80ad9fdd742d195863d57
- hyperpolicy/training_pipeline.py: 990a6825ea5dfb9464511644200880109afac5cc8ac7160136c27b90ba05c081
- tests/hyper_policy_failure_training_test.py: 1b6d52c42577f3bb8e27ee80680430367605606490beef7cd91e9699b9f9ac0d

The integration policy reader contains newer multitask-binding semantics and differs from main; it was tested in its merged form. The unresolved force, sensor-clock, loaded Human, surgical workflow, physical skill and release qualification requirements remain open.
