# Installed CLI checks

From an external workspace, the relocated package's `numi doctor`,
`numi train --help` and native robot-catalog self-check completed; the latter
reported four robots. Doctor found the native executables and an explicitly
supplied MLX environment. These checks do not qualify physics or training.

The original doctor output is retained: it reported readiness even in an
empty workspace without built-in visual source assets. At source revision
`69bcb2a7`, the installed dispatcher additionally reports the missing optional
Window source assets and suggests an authored scene or catalog. This line is
nonfatal because an authored alternative can operate without those inputs; it
does not claim that existing catalogs, models or default visuals have been
qualified. The separate empty-workspace Window failure remains retained.
