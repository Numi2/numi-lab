# Synthetic wound-lip traction — v2 pre-run binding

Frozen after the v2 preregistration and build, and before the CPU-only
preflight or either trajectory.

- Worktree source: `/Users/home/numi-skin-suture-20260930`, branch
  `codex/synthetic-skin-suture-20260930`, HEAD `8ce44aca14f850dc05259c93d5d723e255ae3237`,
  with pre-existing user worktree edits preserved. The transferred source
  snapshot is under
  `/Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-v2-20261004/source`.
- Plan SHA-256:
  `0b39e5aaadb472366721d3f80ff16e721f84c80d3385be7bb277d700a2fe03ba`.
- Source hashes: probe
  `164cd7465d2e79a70997e3ae9b4785327c745b2ac82f32dd674ae259786b0498`;
  CMake `e795e48dff095c50eb01b422e7ee27e6f2150629b86287d950391f37de7a7a47`;
  Matter compiler `c11dee989f25e19031e26f91e2df2b9826049c5f1002e148445bab9b4ce7bd56`;
  runtime `96c6926d54ccb32d5e5fa92ab6292052214fbb9ff0dd06aad4bd074a50abe9e1`;
  surgical coupon `f3e4e2faa4353a364197750c44097e031ffb37850904c725645a7da424ebaa07`.
- Material SHA-256:
  `379945f593f12396b44c7148026f0250344a86278e01b7b26cb5f6d823a51e35`.
- Mac mini: `Mac16,11`, Apple M4 Pro, 12 CPU cores, 24 GiB RAM, macOS 26.6.
  Build: CMake 4.4.1, AppleClang 21.0.0, Xcode 26.6, Release,
  `BUILD_TESTING=ON`, target
  `metalrobo_synthetic_skin_wound_traction_probe`, parallelism 8. Configure
  and build logs are retained in this run directory.
- Probe binary SHA-256:
  `9853b28818619432871a568a26f84a7e49cb8dbeed9835b65d97e56d81a5fd20`.
- Loaded `libmetalrobo.dylib` SHA-256:
  `32d5848cc57c36b3d04462df053cf6cb49d36c70e0a3ff285bbcf95554d6c235`.
- Loaded `NumiMatter.metallib` SHA-256:
  `4b24ba6845caab23d3b51cbc5e16b061e532f0ad0c5a622588f48b16412d2762`.
  The executable's RPATH and `DYLD_PRINT_LIBRARIES` resolve to this run's
  isolated `build/lib/libmetalrobo.dylib`.
- Pre-run check found no traction, suture, or Open Knee process; 38 GiB were
  free. The `numi` dispatcher is unavailable on this host; raw outputs and
  analysis are retained with this native owner run.

The CPU-only preflight command is:

```sh
/Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-v2-20261004/build/bin/metalrobo_synthetic_skin_wound_traction_probe \
  /Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-v2-20261004/preflight-no-output \
  /Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-v2-20261004/source/matter/materials/synthetic_skin_wound.nmatter \
  /Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-v2-20261004/build/matter/shaders/NumiMatter.metallib \
  --validate-only
```

After that preflight passes, run the control in `trial-control/output` with
`--force-per-site-n 0`, then the loaded arm in `trial-loaded/output` with
`--force-per-site-n 0.010`. Each command uses the same material and
`NumiMatter.metallib` listed above.
