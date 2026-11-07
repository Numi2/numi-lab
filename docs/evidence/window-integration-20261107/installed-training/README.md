# Installed training and viewer discovery

The relocated installed package from source commit `9e6e506b` ran `numi train`
from an external workspace with one environment, eight physics steps, one PPO
update, one minibatch, and a matched one-environment selection at the existing
physical episode horizon. The command completed without timeout. The incumbent
was retained; both incumbent and candidate had zero clean-horizon completion
and 12 physical terminations. The entire run directory, generated policies,
optimizer state, rollout, selection details, checksums, native logs, command,
package hashes and timing are retained. This is a bounded execution and package
smoke, not evidence of learned control or physical accuracy.

Installed `numi view --help` and `numi view catalog` also completed from the
external workspace. They exercise CLI discovery only, not opening a viewer or
revealing an observation. Guided training through the Window setup panel and
cross-owner snapshots remain unqualified.
