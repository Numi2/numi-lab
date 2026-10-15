# Source-bound commands 450/452

Exact copies of the two full launcher command vectors are bound to launch-guard records, predictions, and terminal receipts. Prediction argv/environment, case/source/runtime/build paths, input sets, timeout, and command hashes were verified. The runner is referenced by exact hash from retained source142; scientific inputs and source/build trees are not duplicated.

After output-path normalization, probe arguments and all common environment values match. The only removed functional environment variables are `NM_FEM_BASE_CGLS_CAPTURE_NEWTON` and `NM_FEM_BASE_CGLS_CAPTURE_PATH`; common status/operator-capture paths point to each run's own directory. This is command provenance, not a parity result.
