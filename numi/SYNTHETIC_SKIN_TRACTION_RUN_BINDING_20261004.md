# Synthetic wound-lip traction — pre-run binding

This binding was sealed after the plan in
`SYNTHETIC_SKIN_TRACTION_STUDY_20261004.md` and before either trajectory ran.

- Worktree: `/Users/home/numi-skin-suture-20260930`, branch
  `codex/synthetic-skin-suture-20260930`, HEAD `8ce44aca14f850dc05259c93d5d723e255ae3237`,
  with unrelated existing worktree modifications preserved.
- Exact probe source SHA-256:
  `ddafd4a765cdbb7264bac9b397d1395d78c16324f1c1456a699b7ff228aad334`.
- Synthetic material SHA-256:
  `379945f593f12396b44c7148026f0250344a86278e01b7b26cb5f6d823a51e35`.
- Mac mini: `Mac16,11`, Apple M4 Pro, 12 CPU cores, 24 GiB RAM, macOS 26.6;
  CMake 4.4.1, AppleClang 21.0.0, Release build, target
  `metalrobo_synthetic_skin_wound_traction_probe`.
- Probe binary SHA-256:
  `badc9afbb441914029217c8aec635f15e9fb7eec3f6e0740fa2da5375b5e9c72`.
- Loaded `libmetalrobo.dylib` SHA-256:
  `76e9bce7c0578b46c590cfb3c2fbe2b118fbbac0f307a63db97968b8c7b1db2e`.
- Loaded `NumiMatter.metallib` SHA-256:
  `4b24ba6845caab23d3b51cbc5e16b061e532f0ad0c5a622588f48b16412d2762`.
- Both binaries and the material reside in the isolated run directory
  `/Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-20261004`.
  `otool` and `DYLD_PRINT_LIBRARIES` confirmed the executable loads the library
  from this run's own `build/lib`, not a neighboring experiment.
- Pre-run ownership check found no native skin, suture, Open Knee, or traction
  process; 38 GiB were available on the Mac mini volume.
- The `numi` dispatcher and `numi science` notebook are unavailable on that
  host. The retained preregistration, exact runner command, per-arm raw CSVs,
  stdout/stderr, and post-run analysis comprise the local evidence record.

The ordered commands are:

```sh
/Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-20261004/build/bin/metalrobo_synthetic_skin_wound_traction_probe \
  /Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-20261004/trial-control/output \
  /Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-20261004/source/matter/materials/synthetic_skin_wound.nmatter \
  /Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-20261004/build/matter/shaders/NumiMatter.metallib \
  --force-per-site-n 0
```

The loaded-arm command is identical except the output directory is
`trial-loaded/output` and the final value is `0.010`.
