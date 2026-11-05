#!/usr/bin/env python3
"""Reader-versioned 030d/038e adapter over the immutable 014 source-only receipt replay."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
from typing import Any

CORE_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/postpass-005/build_current_fractional_mass_receipt_core_1061.py")
CORE_SHA256 = "a5dcf6fc47a57248edb5c1c0086d879aa705d6dd9bb78c5ea102dd35986cc247"
FLOW_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/postpass-032-seed-033/recorded_flow_contract_013_seed033_receipt.py")
FLOW_SHA256 = "7b09ee8936b8487f77a330b7acd2fdb130c2399e380f3c39aac5a63918eb5b29"
VARIANT_030D = "030d-contiguous-preference-lookup"
VARIANT_038E = "030d-base+033-bounded-sparse-seed"
STATUS_SOURCE_ONLY = "PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN"
RUN_CONTROL_ADAPTER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/postpass-005/adapter_current_regional_1061.py")
RUN_CONTROL_ADAPTER_SHA256 = "08cc0b9870af3f48fcd325fc004374829ef32467e62241275ea03c551c9d34dc"
RUN_CONTROL_VARS = {
    "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256",
    "NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_pinned(path: Path, digest: str, name: str):
    path = path.resolve(strict=True)
    if sha_file(path) != digest:
        raise ValueError(f"pinned {name} source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_control_manifest_017(run, control_path, *, core, flow, legacy_validator=None):
    """Authenticate the opt-in source policy and actual compiled dependency closure."""
    control=core.read_json(control_path)
    if run.get("variant") != "bounded-regional-compensation-1061+level-pruned-034+redundant-collector-cap-fix" or control.get("variant")!=run["variant"]:
        raise core.ReceiptError("unexpected compensation variant")
    if run.get("physical_admission") is not False or run.get("native_steps")!=0:
        raise core.ReceiptError("source control contains native claims")
    p,digest=core.path_sha_file_pin(run["source_control_manifest"],"source control")
    if p!=control_path or digest!=sha_file(p):raise core.ReceiptError("source control pin differs")
    if run["source"]!=control["source"] or run["binary"]!=control["binary"]:
        raise core.ReceiptError("source/binary differs from actual compile")
    for key in ("source","binary","compile_execution","focused_controls","policy"):
        core.path_sha_file_pin(control[key],key)
    for item in control["dependencies"]:core.path_sha_file_pin(item,"compiled dependency")
    compilation=core.read_json(Path(control["compile_execution"]["path"]))
    if compilation.get("exit_code")!=0:raise core.ReceiptError("compilation failed")
    tests=core.read_json(Path(control["focused_controls"]["path"]))
    if len(tests["tests"])!=9 or any(x.get("compile_exit")!=0 or x.get("test_exit")!=0 for x in tests["tests"]):
        raise core.ReceiptError("focused controls are incomplete or failed")
    env={x.split("=",1)[0]:x.split("=",1)[1] for x in run["argv"] if x.startswith("NUMI_H706_FRACTIONAL_") and "=" in x}
    if env["NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256"]!=digest:
        raise core.ReceiptError("actual argv source-control hash differs")
    config,config_digest=core.path_sha_file_pin(run["run_config"],"run config")
    if env["NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256"]!=config_digest:
        raise core.ReceiptError("actual argv config hash differs")
    policy=core.read_json(Path(control["policy"]["path"]))
    # Metadata is part of the authenticated carrier packet, not a mass edit.
    import numpy as np
    with np.load(env["NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ"],allow_pickle=False) as packet:
        if packet['regional_compensation_parent_rows'].tolist()!=[11630,11634,11661,11662,11663,11721]:
            raise core.ReceiptError("source exception scope changed")
        binding_path=Path('/Users/numi/Work/human-integrated-gaps-20261104/joint-boundary-rebuild-64c/compiler/mass-support/receiver-rebind.npz')
        if sha_file(binding_path)!='8eea17f9e21cc1e6cf814da99946f49e56b678d5af9d270b38815d70c4427efe':raise core.ReceiptError('receiver-rebind pin changed')
        with np.load(binding_path,allow_pickle=False) as maps, np.load('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/skin-source-carriers-with-regional-compensation-002.npz',allow_pickle=False) as baseline:
            # targets() uses the previously authenticated old-row receiver mapping.
            offsets=maps['old_to_new_offsets'];children=maps['old_to_new_rows']
            expected=np.unique(np.concatenate([children[offsets[int(i)]:offsets[int(i)+1]] for i in baseline['regional_compensation_child_rows']]))
            if not np.array_equal(packet['regional_compensation_child_rows'],expected):raise core.ReceiptError('compensation receiver set is not exactly remapped old scope')
            if packet['regional_compensation_limits'].tolist()!=baseline['regional_compensation_limits'].tolist():raise core.ReceiptError('mass or distance exception changed')
    out=dict(control)
    out['admission_adapter']={"path":str(RUN_CONTROL_ADAPTER_PATH),"sha256":RUN_CONTROL_ADAPTER_SHA256}
    core.path_sha_file_pin(out['admission_adapter'],'existing identity adapter')
    return out


def parse_runtime_017(run: dict[str, Any], parser):
    """Keep run-control hashes out of the 006 app env while their adapter verifies them."""
    env, args = parser(run)
    present = RUN_CONTROL_VARS & set(env)
    if present and present != RUN_CONTROL_VARS:
        raise ValueError("030d actual argv must carry both separate run-control hashes")
    for key in RUN_CONTROL_VARS:
        env.pop(key, None)
    return env, args


def canonical_receipt_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_receipt_017(run_path: Path, execution_path: Path, stdout_path: Path,
                      config_path: Path, control_path: Path, output_path: Path):
    core = import_pinned(CORE_PATH, CORE_SHA256, "receipt_builder_014")
    flow = import_pinned(FLOW_PATH, FLOW_SHA256, "recorded_flow_013_030d_038e")
    run_path, execution_path = run_path.resolve(strict=True), execution_path.resolve(strict=True)
    stdout_path, config_path = stdout_path.resolve(strict=True), config_path.resolve(strict=True)
    control_path, output_path = control_path.resolve(strict=True), output_path.resolve()
    run, control = core.read_json(run_path), core.read_json(control_path)
    original = core.validate_control_manifest
    original_parse_runtime = core.parse_runtime
    core.validate_control_manifest = lambda r, p: validate_control_manifest_017(
        r, p, core=core, flow=flow, legacy_validator=original)
    core.parse_runtime = lambda r: parse_runtime_017(r, original_parse_runtime)
    try:
        mass = core.build_receipt(run_path, execution_path, stdout_path, config_path, control_path)
    finally:
        core.validate_control_manifest = original
        core.parse_runtime = original_parse_runtime
    projection = None
    if control.get("variant") in {VARIANT_030D, VARIANT_038E}:
        transport = mass.get("solver", {}).get("transport_receipt_sha256")
        if not isinstance(transport, str) or len(transport) != 64:
            raise ValueError("mass replay lacks exact transport receipt SHA")
        digest = hashlib.sha256(canonical_receipt_bytes(mass)).hexdigest()
        # This call validates only the 030d source-control projection. The later
        # 010 full-bundle verifier checks the populated per-role records.
        bundle = {
            "schema": flow.BASE.SCHEMA,
            "unified_mass_receipt": {"path": str(output_path), "sha256": digest},
            "source_control_manifest": run["source_control_manifest"],
            "source_closure": run["source_closure"],
            "paired_solver_run_sha256": transport,
            "roles": {},
        }
        flow.verify_source_control(run, bundle, mass_receipt=mass)
        projection = {
            "schema": "numi.h706.current-fractional-mass-source-projection-017.v1",
            "status": ("PASS_SOURCE_ONLY_038E_STREAMED_READER_CONTROL_PROJECTION" if control.get("variant") == VARIANT_038E else "PASS_SOURCE_ONLY_030D_STREAMED_READER_CONTROL_PROJECTION"),
            "physical_admission": False, "native_steps": 0,
            "run_manifest": {"path": str(run_path), "sha256": sha_file(run_path)},
            "source_control_manifest": {"path": str(control_path), "sha256": sha_file(control_path)},
            "source_closure": run["source_closure"],
            "mass_receipt": {"path": str(output_path), "sha256": digest},
            "receipt_builder_014": {"path": str(CORE_PATH), "sha256": CORE_SHA256},
            "recorded_flow_contract_013": {"path": str(FLOW_PATH), "sha256": FLOW_SHA256},
            "reader_revision": control.get("reader_revision"),
            "reader_build_config": control.get("reader_build_config"),
            "input_reader": control.get("input_reader"),
            "receipt_builder_017": {"path": str(Path(__file__).resolve()), "sha256": sha_file(Path(__file__).resolve())},
            "run_control_compatibility_adapter": {
                "path": str(RUN_CONTROL_ADAPTER_PATH),
                "sha256": RUN_CONTROL_ADAPTER_SHA256,
            },
            "full_recorded_flow_bundle": "NOT_VALIDATED_BY_THIS_SOURCE_PROJECTION_STAGE",
        }
    return mass, projection


def validate_output_paths(output_path: Path, projection_path: Path | None,
                          *, projection_required: bool) -> tuple[Path, Path | None]:
    output = output_path.resolve()
    projection = projection_path.resolve() if projection_path is not None else None
    if output.exists():
        raise ValueError("refusing to overwrite existing mass receipt")
    if projection_required and projection is None:
        raise ValueError("source-controlled variants require a distinct source-projection report path")
    if projection is not None:
        if projection == output:
            raise ValueError("mass receipt and source-projection report paths must differ")
        if projection.exists():
            raise ValueError("refusing to overwrite existing source-projection report")
    return output, projection


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical_receipt_bytes(value))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-manifest", required=True, type=Path)
    p.add_argument("--execution", required=True, type=Path)
    p.add_argument("--stdout-log", required=True, type=Path)
    p.add_argument("--identity-config", required=True, type=Path)
    p.add_argument("--control-manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--source-projection-report", type=Path)
    p.add_argument("--failure-report", type=Path)
    a = p.parse_args()
    try:
        control_preview = json.loads(a.control_manifest.resolve(strict=True).read_text(encoding="utf-8"))
        require_projection = control_preview.get("variant") in {VARIANT_030D, VARIANT_038E}
        out, projection_out = validate_output_paths(
            a.output, a.source_projection_report, projection_required=require_projection)
        mass, projection = build_receipt_017(
            a.run_manifest, a.execution, a.stdout_log, a.identity_config,
            a.control_manifest, out)
        write_json(out, mass)
        if projection is not None:
            if projection_out is None:
                raise ValueError("030d projection output was not predeclared")
            rp = projection_out
            write_json(rp, projection)
            if sha_file(out) != projection["mass_receipt"]["sha256"]:
                raise ValueError("written mass receipt differs from projected receipt digest")
            print(f"recorded_flow_source_projection=PASS path={rp} sha256={sha_file(rp)} physical_admission=0 native_steps=0")
        print(f"fractional_mass_receipt=PASS schema={mass['schema']} path={out} sha256={sha_file(out)} physical_admission=0 native_steps=0")
        return 0
    except Exception as exc:
        failure = a.failure_report or a.output.with_suffix(".failure.json")
        write_json(failure.resolve(), {
            "schema": "numi.h706.current-fractional-mass-receipt-build-failure-017.v1",
            "status": "REJECTED", "physical_admission": False, "native_steps": 0,
            "reason": str(exc),
        })
        print(f"fractional_mass_receipt=REJECTED reason={exc} failure_report={failure.resolve()}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
