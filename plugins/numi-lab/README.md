# Numi Lab plugin

Numi Lab is the local Codex guide to the Apple-native research suite. It helps
Codex discover the owning tool for a task, run supported workflows in a local
environment, and report what the retained evidence actually establishes.

## Start here

Try one of these requests:

- “Show the Numi tools available on this Mac and what I can run next.”
- “Prepare a NumiVivo MD run from my existing system and verify its output.”
- “Evaluate a Numi robot policy on held-out rollouts and report the physical outcomes.”

The skill includes local starting paths for suite discovery, prepared molecular
dynamics, tissue campaign compilation, Human standing diagnostics, robot policy
evaluation, and solver profile inspection. It routes other suite requests to
the relevant owner and its current help. The workflow guide is
[local-workflows.md](skills/numi-lab/references/local-workflows.md).

## Install and use in Codex

Install from the repository marketplace with `codex plugin add
numi-lab@numi-lab`. Start a new Codex thread after installation or an update;
active threads keep their previously loaded skill snapshot. `numi codex status`
checks the installed cache against the source. If the dispatcher is slow,
`codex plugin list --json` still shows the installed plugin and version.

The plugin does not install Numi runtimes. Execution requires the relevant
owner CLI and access to the Mac in the current Codex environment. A cloud or
remote thread must check its selected host rather than assume the Mac is
reachable. If execution is unavailable, Codex should inspect any supplied
files and name the next useful local step.

Native results are reported with their command, runtime identity, artifacts,
measured outcome, and remaining evidence limits. A compiled study is not an
executed study; simulation is not hardware evidence; an electronic energy is
not a reaction rate.

## Maintain the plugin

The compatibility manifest is [plugin.json](.codex-plugin/plugin.json). The
package contains a skill and local assets, with no MCP server or custom UI.
Local installation uses the `numi-lab` repository marketplace.

After changing the package, validate the skill and plugin, update the manifest
cachebuster, reinstall locally, and start a new conversation to load the
updated skill.
