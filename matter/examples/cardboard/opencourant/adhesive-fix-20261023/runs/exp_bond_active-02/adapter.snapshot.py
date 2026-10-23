#!/usr/bin/env python3
"""Bounded external OpenCourant admission runner; does not simulate mechanics.

Run inside the Linux ARM64 guest. Export and collect also work on macOS.
Successful execution is deliberately separate from physical qualification.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

RELEASE = "latest-20261011"
SOURCE = "cb760c206e346eff12161a2268b134ef9d48a4ef"
PACKAGE_SHA256 = "ded9814e2460767353d591c89695ba1f76251291041943ab0ad73070a6c865f7"
TYPE43_PATCH_SHA256 = "3fa4e0f2d2ae360371e3d0a9291fefff2103d9d5c09bbaf10d28cc8a9d6a0488"


def runtime_binaries(runtime: Path, identity: dict, profile: str) -> dict:
    """Pinned installation candidates; modified mechanics has its own identity."""
    if identity.get("package_sha256") != PACKAGE_SHA256 or identity.get("source_commit") != SOURCE:
        raise ValueError("runtime is not bound to the pinned package/source")
    engine = "engine_linuxa64_gf"
    if profile == "arm64-mumps-build":
        build = identity.get("build", {})
        if (identity.get("profile") != profile or build.get("source_commit") != SOURCE
                or build.get("tracked_source_diff") != "" or not build.get("receipt_sha256")):
            raise ValueError("MUMPS build requires unchanged pinned source and a build receipt")
        receipt = runtime.parent.parent / "setup" / "build-receipt.json"
        if sha(receipt) != build["receipt_sha256"]:
            raise ValueError("MUMPS build receipt identity changed")
        engine = "engine_linuxa64_gf_ompi"
    elif profile == "arm64-mumps-type43":
        build = identity.get("build", {})
        setup = runtime.parent.parent / "setup"
        if (identity.get("profile") != profile or build.get("source_commit") != SOURCE
                or build.get("patch_sha256") != TYPE43_PATCH_SHA256
                or sha(setup / "type43-implicit.patch") != TYPE43_PATCH_SHA256
                or sha(setup / "build-receipt.json") != build.get("receipt_sha256")):
            raise ValueError("TYPE43 candidate requires the pinned patch and build receipt")
        receipt = json.loads((setup / "build-receipt.json").read_text())
        if (receipt.get("source_commit") != SOURCE
                or receipt.get("patch_sha256") != TYPE43_PATCH_SHA256
                or receipt.get("engine_sha256") != identity["files"].get("exec/engine_type43")):
            raise ValueError("TYPE43 build receipt does not bind this engine")
        engine = "engine_type43"
    elif profile != "arm64-package" or identity.get("profile", profile) != profile:
        raise ValueError("unknown or mismatched runtime profile")
    return {"starter": local_file(runtime, "exec/starter_linuxa64_gf"),
            "engine": local_file(runtime, f"exec/{engine}")}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inventory(root: Path) -> dict:
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"symlink not admitted: {p}")
        if p.is_file():
            result[str(p.relative_to(root))] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    return result


def local_file(root: Path, name: str) -> Path:
    p = (root / name).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError(f"missing or escaping input: {name}")
    return p


def execute(argv: list[str], cwd: Path, env: dict, timeout: float, label: str) -> dict:
    start = time.monotonic()
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    timed_out = False
    with (cwd / f"{label}.stdout").open("wb") as out, (cwd / f"{label}.stderr").open("wb") as err:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
        write(cwd / f"{label}.process.json", {"pid": p.pid, "argv": argv, "started_utc": stamp})
        try:
            status = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                status = p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                status = p.wait()
    return {"argv": argv, "exit_code": status, "timed_out": timed_out,
            "elapsed_s": time.monotonic() - start, "started_utc": stamp}


def native_termination(text: str) -> dict:
    """A process exit alone is insufficient for a completed solver stage."""
    return {"normal_termination": bool(re.search(r"NORMAL\s+TERMINATION", text, re.I)),
            "error_termination": bool(re.search(r"ERROR\s+TERMINATION|ERROR ID\s*:", text, re.I))}


def continuation(engines: list[str], parent: dict, requested: str) -> None:
    previous = parent.get("completed_engine")
    if previous not in engines or requested not in engines:
        raise ValueError("resume requires recorded and preregistered Engine stages")
    if engines.index(requested) != engines.index(previous) + 1:
        raise ValueError("resume must consume the immediately preceding accepted Engine checkpoint")


def export_cases(args) -> int:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    suites = ["paper", "adhesion"] if args.suite == "all" else [args.suite]
    cases = []
    for suite in suites:
        module = importlib.import_module(f"cardboard_opencourant_{suite}")
        options = {"include_instrumented": True} if suite == "implicit" and args.instrumented else {}
        cases.extend(getattr(module, f"export_{suite}_cases")(out, **options))
    write(out / "suite.json", {"schema": "cardboard.opencourant.suite.v1", "release": RELEASE,
          "source_commit": SOURCE, "cases": cases, "physical_qualification": False})
    print(json.dumps({"output": str(out), "cases": len(cases)}))
    return 0


def run_case(args) -> int:
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise ValueError("This pinned runtime requires native Linux ARM64; use the isolated Lima guest")
    case_path = args.case.resolve()
    spec = json.loads(case_path.read_text())
    source_dir = case_path.parent
    inputs = inventory(source_dir)
    starter = spec["starter"]
    engines = spec["engines"]
    if not isinstance(engines, list) or not engines:
        raise ValueError("engines must be a nonempty ordered filename list")
    for name in [starter, *engines]:
        local_file(source_dir, name)
    runtime = args.runtime.resolve()
    identity_path = runtime.parent.parent / "setup" / "runtime-identity.json"
    identity = json.loads(identity_path.read_text())
    profile = getattr(args, "runtime_profile", "arm64-package")
    binaries = runtime_binaries(runtime, identity, profile)
    for name, digest in identity["files"].items():
        if sha(local_file(runtime, name)) != digest:
            raise ValueError(f"runtime changed since preparation: {name}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Inputs are small admission decks; retain exact bytes, never overwrite a prior attempt.
    for name in inputs:
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / name, dest)
    shutil.copyfile(Path(__file__).resolve(), out / "adapter.snapshot.py")
    parent = None
    if getattr(args, "parent", None):
        parent_path = args.parent.resolve()
        parent = json.loads(parent_path.read_text())
        if not parent.get("execution_complete") or parent.get("case_sha256") != sha(case_path):
            raise ValueError("resume requires a completed parent with the same case identity")
        if not parent.get("checkpoints"):
            raise ValueError("parent has no retained restart checkpoint")
        if parent.get("binary_sha256") != {k: sha(v) for k, v in binaries.items()}:
            raise ValueError("resume binary identity differs")
        if parent.get("runtime_identity_sha256") != sha(identity_path):
            raise ValueError("resume runtime identity differs")
        continuation(engines, parent, args.engine)
        previous_stem = Path(parent["completed_engine"]).stem
        for name, item in parent["checkpoints"].items():
            p = local_file(parent_path.parent, name)
            if sha(p) != item["sha256"]:
                raise ValueError(f"restart hash mismatch: {name}")
            if Path(name).name.startswith(previous_stem + "_"):
                shutil.copyfile(p, out / name)
        if not list(out.glob(previous_stem + "_*.rst")):
            raise ValueError("missing immediately preceding Engine restart")
        engines = [args.engine]
        local_file(out, args.engine)
    elif getattr(args, "through_engine", None):
        if args.through_engine not in engines:
            raise ValueError("through-engine must be a preregistered Engine stage")
        engines = engines[:engines.index(args.through_engine) + 1]
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "USER", "LANG", "TMPDIR")}
    env.update({"RAD_CFG_PATH": str(runtime / "hm_cfg_files"),
                "LD_LIBRARY_PATH": ":".join(str(runtime / p) for p in
                     ("extlib/hm_reader/linuxa64", "extlib/h3d/lib/linuxa64")),
                "OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1"})
    loaded_libraries = {}
    for binary in binaries.values():
        ldd = subprocess.run(["ldd", str(binary)], env=env, capture_output=True, text=True, check=True)
        if "not found" in ldd.stdout:
            raise ValueError(f"unresolved runtime library: {ldd.stdout}")
        for name in re.findall(r"(?:=>\s*|^\s*)(/\S+)", ldd.stdout, re.M):
            library = Path(name)
            loaded_libraries[str(library)] = sha(library)
    start = time.monotonic()
    receipt = {"schema": "cardboard.opencourant.run.v1", "release": RELEASE, "source_commit": SOURCE,
               "runtime_profile": profile,
               "case_id": spec["id"], "case_sha256": sha(case_path), "inputs": inputs,
               "binary_sha256": {k: sha(v) for k, v in binaries.items()}, "environment": env,
               "runtime_identity_sha256": sha(identity_path),
               "adapter_sha256": sha(Path(__file__).resolve()),
               "loaded_library_sha256": loaded_libraries,
               "host": {"system": platform.system(), "machine": platform.machine()},
               "parent_receipt_sha256": sha(args.parent.resolve()) if parent else None,
               "commands": [], "execution_complete": False, "physical_qualification": False,
               "memory_limit_bytes": 16 * 1024**3}
    write(out / "run.json", receipt)
    stages = ([] if parent else [("starter", starter)]) + [("engine", e) for e in engines]
    for i, (kind, name) in enumerate(stages):
        argv = ["/usr/bin/time", "-v", str(binaries[kind]), "-i", name]
        if kind == "starter":
            argv += ["-np", "1"]
        else:
            argv += ["-nt", str(args.threads)]
        result = execute(argv, out, env, args.timeout, f"{i:02d}-{kind}")
        result.update(native_termination((out / f"{i:02d}-{kind}.stdout").read_text(errors="replace")))
        receipt["commands"].append(result)
        write(out / "run.json", receipt)
        if result["exit_code"] or result["timed_out"] or result["error_termination"] or not result["normal_termination"]:
            break
        if kind == "engine":
            receipt["completed_engine"] = name
    else:
        receipt["execution_complete"] = True
    receipt["elapsed_s"] = time.monotonic() - start
    receipt["case_sequence_complete"] = receipt.get("completed_engine") == spec["engines"][-1]
    receipt["outputs"] = {k: v for k, v in inventory(out).items() if k != "run.json"}
    receipt["checkpoints"] = {k: v for k, v in receipt["outputs"].items() if k.endswith(".rst")}
    write(out / "run.json", receipt)
    print(json.dumps({"receipt": str(out / "run.json"), "execution_complete": receipt["execution_complete"],
                      "elapsed_s": receipt["elapsed_s"], "physical_qualification": False}))
    return 0 if receipt["execution_complete"] else 2


def collect(args) -> int:
    path = args.receipt.resolve()
    receipt = json.loads(path.read_text())
    root = path.parent
    changed = [name for name, item in receipt["outputs"].items()
               if not (root / name).is_file() or sha(root / name) != item["sha256"]]
    peaks = []
    terminations = []
    for p in sorted(root.glob("*.stderr")):
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", p.read_text(errors="replace"))
        if match:
            peaks.append(int(match[1]) * 1024)
    for p in sorted(root.glob("*_000[1-9].out")):
        text = p.read_text(errors="replace")
        terminations.append({"file": p.name, **native_termination(text), "tail": text[-3000:]})
    report = {"schema": "cardboard.opencourant.collection.v1", "receipt_sha256": sha(path),
              "execution_complete": receipt["execution_complete"], "output_identity_valid": not changed,
              "changed_outputs": changed, "max_solver_rss_bytes": max(peaks) if peaks else None,
              "native_termination": terminations, "physical_qualification": False,
              "limitation": "Process completion and output hashes do not establish paper, interface, assembly or convergence gates."}
    write(root / "collection.json", report)
    print(json.dumps(report))
    return 0 if not changed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--suite", choices=("paper", "adhesion", "implicit", "all"), default="all")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--instrumented", action="store_true", help="Also export instrumented implicit controls")
    p.set_defaults(func=export_cases)
    for name in ("run", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--case", type=Path, required=True)
        p.add_argument("--runtime", type=Path, required=True)
        p.add_argument("--runtime-profile", choices=("arm64-package", "arm64-mumps-build", "arm64-mumps-type43"),
                       default="arm64-package")
        p.add_argument("--output", type=Path, required=True)
        if name == "run":
            p.add_argument("--through-engine", help="Stop after this declared stage, retaining later inputs for resume")
        p.add_argument("--timeout", type=float, default=120)
        p.add_argument("--threads", type=int, default=1)
        if name == "resume":
            p.add_argument("--parent", type=Path, required=True)
            p.add_argument("--engine", required=True)
        p.set_defaults(func=run_case)
    p = sub.add_parser("collect")
    p.add_argument("--receipt", type=Path, required=True)
    p.set_defaults(func=collect)
    args = parser.parse_args()
    if hasattr(args, "timeout") and (not 0 < args.timeout <= 1800 or not 1 <= args.threads <= 8):
        parser.error("admission timeout must be (0,1800] seconds and threads 1..8")
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
