# Installed selector cache repair: non-humanoid smoke

At source revision `4aee60dc1b607b179dd2bb93278ee11f31a15bcc`, an isolated
package was configured and installed. The installed selector SHA-256 matched
source, and the installed WorldPack CLI help ran successfully. `plan.json`
pins package, executable, shader, input and fixture identities.

The synthetic authored Franka WorldPack ran one training update and matched
selection on M6, 2026-11-07 23:54:44.058–23:54:48.948 UTC. Exit was zero,
the incumbent was selected, policy and evidence hashes were bound, and input
hashes were unchanged. The humanoid screen correctly remained unknown, with
auto-load refused, including modified and missing identity checks. No run
lock or matching worker remained. The retained run includes the full output,
state traces, evidence, copied mutation cases, scripts, and build logs.

This smoke exercises the installed workflow and revised selector. The direct
same-path input replacement and unresolved-dependency behavior were tested in
the separate CPU regression at the same source revision; this native smoke
does not independently validate cache invalidation, physical accuracy,
learned control, replay, visual UI, or performance. Input closure is not
established for multi-file inputs, and concurrent dependency changes and
shared-library or environment identity are not covered by cache identity.

Correction: `metalrobo_world_pack --help` was interpreted as an output path,
not a help request. It established that the executable loaded and generated a
pack, not that help worked. The later observation is retained in
`../clean-sparse-package-20261108/`.
