# Merged policy-evaluation native check

The combined integration revision `92489f5c93eecfcc9b8bd7004c075323a00881e4` contains published main `2d07a4497` as an ancestor. It keeps integration's existing initial-reset behavior and newer window and task paths; it does not claim the integration branch is a finished release.

The native authored-state, reset-boundary, timeout, non-timeout and repeat checks passed with zero actions and a freshly generated same-runtime zero policy. Input and runtime hashes and complete outputs are retained. The earlier two-environment one-update synthetic Franka training/selection run is recorded in `../uninterrupted-policy-eval-20261108/`; it was executed before this merge on the equivalent integrated rollout and selector sources. No task skill, contact accuracy or timing conclusion follows. The fine P2 contact reference was running concurrently.
