# Fresh build from a sparse source worktree

A new detached worktree at `d9db7f9f30f75c9d54432fee2ad89d0e8f3ef26d`
was configured in a previously nonexistent Ninja build directory, and its
installable targets were compiled from source without reusing prior build
outputs. The source worktree was clean and shared the parent Git object store.
It was a blobless sparse partial checkout: 117,191 of 119,164 tracked paths
were skipped. This is not a clean full checkout or an offline clone test.

At inspection, source/build/installed package occupied 80,036 / 222,312 /
117,228 KiB respectively, with 18,402,380 KiB available. These are the
observed sizes on this APFS host, not guaranteed minimum storage. Toolchain
and Python environment details are retained in `reproduction-environment.txt`.
CMake selected Python 3.15 and disabled its NumPy/SciPy tests; the training
smoke used an existing Python 3.13 MLX/NumPy environment. Metal was available
through a separately installed toolchain while Command Line Tools was active.
Default Window visual assets remain missing.

The first install failed because the initial explicit build list omitted
Matter's installable runtime. Its full output and 21 MB partial prefix remain
preserved. Building the required Matter targets and installing into a new
prefix succeeded; both logs are retained. The first CPU helper compile also
failed because its evidence source was excluded by sparse patterns; the helper
was then extracted directly from the exact Git revision. The status CTest,
33 Python tests, installed selector producer/Swift consumer checks, and
installed WorldPack inspection passed. The Episode compiler usage command
returned 2. An initial wrapper for the successful inspection hit zsh's
read-only `status` variable; the explicit bash retry passed.

A non-G1 synthetic Franka WorldPack then completed one eight-step, one-update
native training/selection run on M6 at 00:06:30.110–00:06:34.642 UTC on
2026-11-08. Input hashes were unchanged, incumbent retained, and no lock or
worker remained. Metadata contained the world hash. Changed/missing/restored
world checks remained unknown and refused auto-load; the separate synthetic
CPU producer-consumer check demonstrates the positive-to-unknown transition.
The run and hashes are retained. It is not physical, learned-control, replay,
UI or performance qualification.

Correction to earlier records: `metalrobo_world_pack --help` is parsed as an
output filename and writes a WorldPack; it is not a help command. The invoked
executable did load, but earlier statements calling this a successful help
check were wrong. The resulting `/Users/numi/--help` file was not removed;
its observed bytes and SHA are retained here. The genuine read-only inspection
used `--inspect` on the pinned fixture. Dynamic libraries, Python packages,
OS/environment and concurrent input changes remain outside complete identity.
