# Numi Lab plugin

Numi Lab is a skills-only guide to the Apple-native research suite. It helps
ChatGPT and Codex discover the owning tool for a task, run supported local
workflows when execution is available, and report what the retained evidence
actually establishes.

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

## Execution access

The plugin does not install Numi runtimes or connect ChatGPT on the web to a
Mac. Local execution requires the relevant owner CLI and session access to the
machine. If those are missing, the assistant should say so, inspect any files
the user has supplied, and give the next useful local step.

Native results are reported with their command, runtime identity, artifacts,
measured outcome, and remaining evidence limits. A compiled study is not an
executed study; simulation is not hardware evidence; an electronic energy is
not a reaction rate.

## Package and release

The compatibility manifest is [plugin.json](.codex-plugin/plugin.json). The
package contains a skill and local assets, with no MCP server or custom UI.
Local installation uses the `numi-lab` repository marketplace. A public
submission uploads a ZIP of this plugin directory through the
[OpenAI plugin submission flow](https://developers.openai.com/plugins/deploy/submission).

After changing the package, validate the skill and plugin, update the manifest
cachebuster, reinstall locally, and start a new conversation to load the
updated skill.
