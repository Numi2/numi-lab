# Tiled cohesive parity fixture setup overlay

This isolated checker-only overlay is based on frozen source 082. It repairs
the standalone `--cohesive-tiled-parity` fixture setup after native-275 failed
in the legacy arm before tiled scratch was exercised. The failure was a strict
certificate miss at Newton 8 (`1.16402013e-7` vs unchanged `1e-7`, FGMRES 64).

When that mode is selected, the checker now owns and reports the already-used
accurate-paper setup and extended solver budget for both legacy and tiled arms:
accurate paper projection, Newton 32, FGMRES 256, restart 16, and 24 line-search
trials. The relative residual tolerance remains `1e-7`; no physical equation,
acceptance gate, or production solver policy changes. Explicit user-supplied
budget/paper modifiers remain disallowed for this fixture to keep the protocol
deterministic. Other checker modes retain their existing defaults.

The parity fixture still compares the legacy and tiled implementations under
the same material, history, and solver configuration. This setup correction
does not itself qualify the tiled path; a fresh source-bound native run is
required. Failed native-275 remains preserved.

Base snapshot: `snapshots/source-tiled-cohesive-and-portable-coupled-check-082`
Base checker SHA-256: `b7d0fa1a777c0f28438873bd5b84825d5bbc3bc2685c2e5052cbf20b33d2de1b`
Patched checker SHA-256: `6c38e2919097786a2b9e3536bdde428d906009fffb63f29240fa5fcf149b12ed`

No shared source was changed and no GPU run was performed for this overlay.
