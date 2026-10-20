# Reference contact expansion 039: evidence note

Status snapshot: 2026-10-20. This note separates standalone geometry checks, the short 039f native rollback/replay validation, and two incomplete native continuations. A later global-plane correction is being prepared separately and is not part of the tested 039 source snapshot.

## Tested source snapshot

The 039 snapshot is based on `4642668b37f385ee7dde3d9fdde097fc0f5d9ab2`. At the time of the source audit, all 33 files listed in the 039f preparation provenance and all 24 files listed in the default32 native provenance matched their recorded SHA-256 values. The captured audit is point-in-time and does not attest to later working-tree edits. The 039f native provenance's 24-file source set is contained in the 33-file preparation set with matching recorded hashes; details are in [source-hash-audit-039f.json](source-hash-audit-039f.json).

The tracked source delta and full source-hash list are preserved in [tested-source-039/](tested-source-039/). Patch SHA-256: `73a3f128df11b8498199b5b024ffa1d00b21c0fc708ecfad2f1215ca3839baf4`. The patch passed `git apply --check` and reconstruction in a temporary detached worktree, recreating all 33 frozen 039f source hashes. The tracked replay helper is captured as a separately identified additional tool file. Modified evidence prose was excluded from the patch.

## Standalone geometry checks

The later diagnostic, repeat, explicit-flags, and original/strict-flag comparisons passed. The explicit-flags result reports 130 geometry rows, two direct exact VF JVP cases, and zero physical transactions. The 130-row initial and diagnostic GPU payloads are byte-identical (SHA-256 `2b07daba8b3ed156d9d684956102ba6c9ef8603f3965cfd819285b85fca91535`). The first host-side oracle mismatch remains unexplained; its exact source and executable were not independently retained. Later passing runs do not identify its cause. See [geometry-check/README.md](geometry-check/README.md), [geometry-check/result-summary.json](geometry-check/result-summary.json), and the command arrays in [geometry-check/build-command-flags.json](geometry-check/build-command-flags.json) and [geometry-check/oracle-comparison.json](geometry-check/oracle-comparison.json). This is standalone geometry and arithmetic evidence, not integrated simulation evidence.

## Native runs

039f unified verification passed, including a two-step mechanics-only native rollback/replay validation at 125 microseconds per step (0.25 ms total). The native run exited 0 after two accepted CSV rows. The terminal audit reports 26 anatomical regions, 28 registered ligament force rows, and zero legacy reduced fascia ligament rows. Root assistance is false in the logged terminal line; presentation qualification remains pending. See [verification-039f/summary.json](verification-039f/summary.json), [verification-039f/native/invocation.json](verification-039f/native/invocation.json), and [verification-039f/native/provenance.json](verification-039f/native/provenance.json).

The default mechanics-only continuation requested 32 steps (4 ms) and accepted 25 steps (3.125 ms) before failing. It exited 1 after 540.186 seconds at root 25, tick 3, slot 1990: Matter code 6, object 9 / failing index 1990. The exact diagnostics and owner string are retained in [native-default32/summary.json](native-default32/summary.json). This run did not complete the requested horizon. No cause is assigned here.

The weighted-support two-step attempt exited 1 after 10.590182 seconds, before any physical step advanced; its coupled CSV has zero data rows. The runtime rejected initialization with “weighted skin support is limited to fresh released authored-pose initialization.” The root-side review identified a configuration mismatch: the guard expected per-contact bed facets while the retained support payload uses a flat global plane. This is an initialization rejection, not a mechanical response result. Its compact package is [native-weighted2/](native-weighted2/), including the storage-deduplication receipt; no packed output was copied. A follow-on correction is outside this snapshot and has no result recorded here.

## Scope and limits

These results support standalone numerical checks and a short mechanics-only rollback/replay validation. They do not establish sustained support, a completed 32-step continuation, physiological behavior, clinical validity, or production qualification. The first geometry host discrepancy remains open.

## Reproduction commands

From the repository root, rebuild and run the standalone checker:

```sh
/opt/homebrew/bin/cmake --build build-integrated-validation-20261019/profile-build-039 --target numi-matter-cardboard-contact-geometry-check --verbose
build-integrated-validation-20261019/profile-build-039/matter/numi-matter-cardboard-contact-geometry-check
build-integrated-validation-20261019/profile-build-039/matter/numi-matter-cardboard-contact-geometry-check --surface-sort-only build-integrated-validation-20261019/profile-build-039/matter/shaders/NumiMatter.metallib
```

The exact post-failure original-flag and strict-flag compiler/run command arrays are in geometry-check/oracle-comparison.json. Native argv, cwd, and environment are recorded in each run's invocation.json.

For an exact default32 replay, reconstruct the captured source delta in a fresh detached worktree and link the already-tested profile build. The commands below verify the captured source list and the exact executable, metallib, and suite-library hashes before the repository replay helper writes to a unique, nonexistent destination:

```sh
REPO=/Users/numi/numi-human-aortic-contact-opt-20261019
EVIDENCE=$REPO/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/reference-contact-expansion-039
BASE=4642668b37f385ee7dde3d9fdde097fc0f5d9ab2
RECON=$(mktemp -d /tmp/numi-039-source.XXXXXX)
rmdir "$RECON"
git -C "$REPO" worktree add --detach "$RECON" "$BASE"
git -C "$RECON" apply "$EVIDENCE/tested-source-039/tested-candidate.patch"
mkdir -p "$RECON/build-integrated-validation-20261019"
ln -s "$REPO/build-integrated-validation-20261019/profile-build-039" "$RECON/build-integrated-validation-20261019/profile-build-039"
python3 - "$RECON" "$EVIDENCE/native-default32/invocation.json" "$EVIDENCE" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys, tempfile, uuid
root, invocation_path, evidence = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest = json.loads((evidence / "tested-source-039/source-hashes.json").read_text())
for row in manifest["frozen_039f_sources"]:
    assert sha(root / row["path"]) == row["tested_039f_sha256"], row["path"]
for row in manifest["039f_native_runtime_sources"]:
    assert sha(root / row["path"]) == row["native_tested_sha256"], row["path"]
for row in manifest["additional_tracked_tool_files"]:
    assert sha(root / row["path"]) == row["captured_current_sha256"], row["path"]
assert sha(evidence / "tested-source-039/tested-candidate.patch") == manifest["patch_sha256"]
identity = json.loads((evidence / "verification-039f/native/provenance.json").read_text())
inv = json.loads(invocation_path.read_text())
binary = pathlib.Path(inv["argv"][0])
profile = binary.parents[1]
assert sha(binary) == identity["executable_sha256"]
assert sha(profile / "matter/shaders/NumiMatter.metallib") == identity["metallib_sha256"]
assert sha(profile / "lib/libmetalrobo.dylib") == identity["suite_library_sha256"]
destination = pathlib.Path(tempfile.gettempdir()) / ("numi-039-replay-" + uuid.uuid4().hex)
assert not destination.exists()
script = root / "matter/tools/evidence/human-resting-20261005/supported-skin-ligaments-20261020/reproduction.py"
os.environ.pop("NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER", None)
steps = inv["argv"][inv["argv"].index("--muscle-step-count") + 1]
raise SystemExit(subprocess.run([sys.executable, str(script), str(invocation_path), str(destination), "--steps", steps]).returncode)
PY
```

To replay 039f's rollback fixture, use its invocation.json and add `--rollback` (two steps). To replay the weighted attempt, use native-weighted2/invocation.json (two steps). The helper requires a fresh destination; keep each result in a new directory. The captured 039 source patch and binary identities must match before execution.
