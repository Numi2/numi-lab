# Reproduce the bounded admission

This is a failed backend admission, not a box simulation. Read the machine result
before scheduling anything. Five executed case directories contain the exact
Starter/Engine decks and adapter snapshots. Each `run.json` contains executed
argv, environment, input hashes, loaded-library hashes and output hashes.

## Recheck retained evidence locally

From the repository root:

```sh
python3 matter/examples/cardboard/opencourant/evidence-20261023/analyze.py
python3 -m unittest discover -s matter/tools -p 'test_cardboard_opencourant*.py'
```

To recompute the retained elastic-control measurement:

```sh
python3 matter/tools/cardboard_opencourant_measure.py \
  matter/examples/cardboard/opencourant/evidence-20261023/runs/implicit-elastic-instrumented-01 \
  --conversion-receipt matter/examples/cardboard/opencourant/evidence-20261023/post/implicit-elastic-shortname-02/conversion.json \
  --output matter/examples/cardboard/opencourant/evidence-20261023/post/elastic-measurement.json
```

`SHA256SUMS.json` seals this retained package after collection. Verification of
hashes establishes file integrity, not mechanics correctness.

## Retained executable environment

On `macmini`, the task root is `/Users/n/cardboard-opencourant-20261023` and
the Lima guest is `cardboard-mumps`. The guest configuration is `setup/lima.yaml`.
The guest is stopped after this admission; start it explicitly before a new run.
Only this task directory is writable through the guest mount. The runtime is
`runtime/OpenCourant`, with its manifests in `setup/`. Runtime and dependency
hashes must match before reusing a checkpoint. Retain old run directories;
use a fresh output directory for every attempt.

The current binary was built from clean upstream source at
`cb760c206e346eff12161a2268b134ef9d48a4ef`. The final source/build tree is on the
guest filesystem at `/home/n.guest/cardboard-upstream-20261023`, bind-mounted
over the guest's task-root `upstream/` path during the build. The host's
`upstream/` directory retains the separate source copy; its interrupted CMake
cache was retired after verification. A VM restart does
not preserve that bind mount; running the existing Engine does not need it.

Build preparation installed `build-essential gfortran cmake libopenmpi-dev
openmpi-bin libmumps-dev libscalapack-openmpi-dev liblapack-dev libtmglib-dev
libblas-dev python3-dev git time curl ca-certificates` in the Ubuntu 24.04 ARM64
guest. `setup/dpkg-packages.txt` records actual versions; unpinned future apt
packages do not reproduce the same binary identity. `gdb` was installed only for
the retained crash replay.

The package from the prior admission was cloned, preserving its Starter and
support libraries. Its original identity was copied to
`setup/packaged-runtime-identity.json` before adding the new Engine. For a fresh
installation, use the exact release asset and digest from that manifest, with
the image digest in `setup/lima.yaml`. Then run the retained
`matter/tools/build_cardboard_opencourant_mumps.sh TASK_ROOT` inside the guest.
It deliberately refuses to overwrite a completed build receipt. Build logs
record the failed dependency scan, interrupted shared-filesystem build and
successful native-filesystem continuation. The source remained unchanged.

## Export and execute one new control

Export on the local checkout and copy the resulting case into the mounted task
directory. Output directories must not already exist:

```sh
python3 matter/tools/cardboard_opencourant.py export --suite implicit \
  --instrumented --output /tmp/cardboard-implicit-replay
```

Inside the retained guest, the run command shape is:

```sh
python3 /Users/n/cardboard-opencourant-20261023/tools/cardboard_opencourant.py run \
  --case /Users/n/cardboard-opencourant-20261023/cases-implicit/implicit-elastic-nonlinear-force-control-instrumented \
  --runtime /Users/n/cardboard-opencourant-20261023/runtime/OpenCourant \
  --runtime-profile arm64-mumps-build \
  --output /Users/n/cardboard-opencourant-20261023/runs/NEW-UNUSED-ID \
  --timeout 120 --threads 1
```

Use `limactl shell cardboard-mumps -- bash -lc 'COMMAND'` from the remote macOS
host, with `/opt/homebrew/bin` on PATH. The full executed commands for each
retained attempt are authoritative; this example is not a new launch request.

For conversion, copy the native T01 and its `_TITLES` sidecar byte-for-byte to
`probeT01` and `probeT01_TITLES` in a fresh directory. Run the pinned
`th_to_csv_linuxa64 probeT01` there. It writes `probeT01.csv`; stdout is diagnostic
text. The successful conversion receipt records input, alias, title, converter
and output hashes. The original long-path failure remains under `post/`.

The measurement command takes the retained native run plus this conversion
receipt; `python3 matter/tools/cardboard_opencourant_measure.py --help` shows
its exact interface. No missing title, generic channel name or guessed force
unit may stand in for the measured fixture.

Resume/activation/native history preservation remains unqualified. Do not infer
it from the existence of `.rst` files or adapter integrity tests. Successful
evidence verification does not satisfy the full-box admission gates.
