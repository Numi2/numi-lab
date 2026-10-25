#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, os, subprocess, time
E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
job = E / "connected-skin-parent-mass-transfer-067/source-contribution-export-072"
run = job / "run-003/source-preflight"
run.mkdir(parents=True, exist_ok=True)
base_manifest_path = E / "bone-clearance-fit-038-exact-bank-export-039/run-002/base-launch-manifest.json"
working_manifest_path = E / "connected-skin-parent-mass-transfer-067/compiled-cell-export-068/run-004/source-preflight/export-launch-manifest.json"
base = json.loads(base_manifest_path.read_text())
working = json.loads(working_manifest_path.read_text())
argv = list(working["argv"])
argv[0] = str(job / "build-001/numi-human-native")
argv[4] = str(run)
env = os.environ.copy()
env.update(base["environment"])
env["NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV"] = str(run / "initial-skin-fem-nodes.csv")
env["NUMI_HUMAN_INITIAL_SKIN_FEM_COMPILED_NODES_CSV"] = str(run / "compiled-skin-fem-nodes.csv")
env["NUMI_HUMAN_INITIAL_SKIN_FEM_TETS_CSV"] = str(run / "compiled-skin-fem-tets.csv")
env["NUMI_HUMAN_SKIN_FAT_SOURCE_CONTRIBUTIONS_CSV"] = str(run / "premerge-layer-contributions.csv")
env["NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP"] = str(run / "tissue-mass-reference.txt")
env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"] = str(run / "common-field-failure.json")
env["NUMI_HUMAN_SOURCE_ONLY_COMPILED_TETS"] = "1"
for key in ("NUMI_HUMAN_INITIAL_SKIN_BONE_ALL_BANK_CSV",
            "NUMI_HUMAN_SKIN_REFERENCE_DUMP",
            "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP",
            "NM_FEM_WRITE_NATIVE_TANGENT_SOURCE",
            "NM_FEM_WRITE_NATIVE_TANGENT_SOURCE_ONLY"):
    env.pop(key, None)
env["NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE"] = "0"
manifest = {
    "argv": argv, "cwd": base["cwd"],
    "environment": {k: env[k] for k in sorted(base["environment"]) if k in env},
    "added_environment": {k: env[k] for k in sorted(env)
        if k.startswith("NUMI_HUMAN_") and k not in base["environment"]},
    "removed_environment_keys": sorted(k for k in base["environment"] if k not in env),
    "base_manifest": str(base_manifest_path),
    "base_manifest_sha256": hashlib.sha256(base_manifest_path.read_bytes()).hexdigest(),
    "patched_binary_sha256": hashlib.sha256(Path(argv[0]).read_bytes()).hexdigest(),
    "scope": "full frozen 039 source-preflight environment; source contribution plus compiled FEM dump; compiled-tet branch exits with _Exit(0) before a simulation step"
}
manifest_path = run / "launch-manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
start = time.time()
with (run / "stdout.txt").open("wb") as out, (run / "stderr.txt").open("wb") as err:
    result = subprocess.run(argv, cwd=base["cwd"], env=env, stdout=out, stderr=err, check=False)
summary = {"returncode": result.returncode, "elapsed_seconds": time.time()-start,
    "launch_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "outputs": {}}
for name in ("premerge-layer-contributions.csv", "initial-skin-fem-nodes.csv", "compiled-skin-fem-nodes.csv",
             "compiled-skin-fem-tets.csv", "tissue-mass-reference.txt", "stdout.txt", "stderr.txt"):
    p = run / name
    if p.exists():
        summary["outputs"][name] = {"bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
summary_path = run / "run-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if result.returncode != 0:
    print("STDERR_TAIL:")
    print("\n".join((run / "stderr.txt").read_text(errors="replace").splitlines()[-12:]))
