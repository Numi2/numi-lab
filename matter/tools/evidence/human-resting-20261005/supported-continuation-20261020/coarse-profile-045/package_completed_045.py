#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
REPO = EVIDENCE.parents[5]
MAX_SMALL_ARTIFACT = 2 * 1024 * 1024
MAX_CSV = 256 * 1024
SKIP_SUFFIXES = {".mrvpack", ".metallib", ".dylib", ".a", ".o", ".obj", ".nhrigid", ".nhmyo", ".nhbones", ".nhskin", ".nhtissue", ".nhcnt", ".nhfascia", ".nheq", ".nhtendon"}
SKIP_NAMES = {"core", "core.0", "core.1", "resting-airway-anchor-map.json", "resting-ocular-sclera-window.json"}
SELECT_TOKENS = ("pose_joint_ranges=passed", "tendon_payload=", "resting_fascia_regional_admission=accepted", "resting_fascia_fem=admitted", "whole_body_accepted_fem_skin_transfer=prepared", "resting_skin_fat_whole_body_preparation=accepted", "resting_fascia_reference_mode=verified", "resting_tissue_mass_partition=conserved", "resting_tissue_shared_runtime=bound", "shared_tissue_late_reject_retry=passed", "resting_FEM_terminal_audit=", "resting_fascia_fem_terminal_audit=", "resting_integrated_body=completed", "resting_common_initialization=accepted_t0", "vascular_initial_guess=")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def small_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def selected_lines(path: Path, destination: Path):
    kept=[]
    with path.open(errors="replace") as stream:
        for index,line in enumerate(stream,1):
            if any(token in line for token in SELECT_TOKENS):
                excerpt=line.rstrip()
                if len(excerpt)>6000: excerpt=excerpt[:6000]+f"... [truncated source line length={len(line)}]\n"
                kept.append(f"{index}:{excerpt}\n")
    destination.write_text("".join(kept))

def main():
    parser=argparse.ArgumentParser(description="Compact a completed 045 profile result without touching root-owned evidence")
    parser.add_argument("--source-dir", required=True, help="Completed run directory under build-integrated-validation-20261019")
    parser.add_argument("--output-dir", default=str(EVIDENCE / "native-integrated-045"), help="New child directory under this evidence directory")
    args=parser.parse_args()
    source=Path(args.source_dir).resolve()
    output=Path(args.output_dir).resolve()
    allowed=(REPO / "build-integrated-validation-20261019").resolve()
    if not source.is_relative_to(allowed):
        raise SystemExit("source must be under build-integrated-validation-20261019")
    if output.parent != EVIDENCE.resolve():
        raise SystemExit("output must be a new direct child of the 043 evidence directory")
    if output.exists():
        raise SystemExit("refusing to overwrite an existing evidence child")
    native_dir=source if (source / "execution.json").is_file() else source / "native"
    execution=native_dir / "execution.json"
    invocation=native_dir / "invocation.json"
    provenance=native_dir / "provenance.json"
    if not execution.is_file() or not invocation.is_file() or not provenance.is_file():
        raise SystemExit("completed source needs execution, invocation, and provenance JSON")
    execution_data=small_json(execution)
    invocation_data=small_json(invocation)
    provenance_data=small_json(provenance)
    if not isinstance(execution_data.get("exit_code"),int):
        raise SystemExit("execution record is incomplete")
    output.mkdir(parents=True)
    copied=[]
    omitted=[]
    raw_hashes={}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel=path.relative_to(source)
        if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES or (path.stat().st_mode & 0o111):
            omitted.append({"path":rel.as_posix(),"size_bytes":path.stat().st_size,"reason":"binary, input payload, or pack excluded"})
            continue
        if path.name in {"stdout.log","stderr.log"} and path.stat().st_size > MAX_SMALL_ARTIFACT:
            raw_hashes[rel.as_posix()]={"size_bytes":path.stat().st_size,"sha256":sha(path),"copied":False}
            excerpt=output/(rel.parent / (path.stem + ".selected.log"))
            excerpt.parent.mkdir(parents=True,exist_ok=True)
            selected_lines(path,excerpt)
            copied.append({"source":str(path),"destination":excerpt.relative_to(output).as_posix(),"size_bytes":excerpt.stat().st_size,"sha256":sha(excerpt),"selection":"filtered diagnostic markers; raw log omitted due size"})
            continue
        if path.stat().st_size > (MAX_CSV if path.suffix.lower()==".csv" else MAX_SMALL_ARTIFACT):
            omitted.append({"path":rel.as_posix(),"size_bytes":path.stat().st_size,"reason":"large raw table or geometry artifact excluded"})
            raw_hashes[rel.as_posix()]={"size_bytes":path.stat().st_size,"sha256":None,"copied":False}
            continue
        target=output/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        digest=sha(target)
        copied.append({"source":str(path),"destination":rel.as_posix(),"size_bytes":target.stat().st_size,"sha256":digest})
        raw_hashes[rel.as_posix()]={"size_bytes":path.stat().st_size,"sha256":sha(path),"copied":True}
    context=[]
    for name in ("TRIAL.md","TRIAL.sha256","template.json","native.log"):
        path=EVIDENCE/name
        if not path.is_file():
            raise SystemExit("required 045 context file is missing: " + path.name)
        target=output/"context"/path.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        context.append({"source":str(path),"destination":target.relative_to(output).as_posix(),"size_bytes":target.stat().st_size,"sha256":sha(target)})
    argv=invocation_data.get("argv",[])
    def flag_value(flag):
        try: return argv[argv.index(flag)+1]
        except (ValueError,IndexError): return None
    coupled=output/((Path("output") if native_dir==source else Path("native/output")) / "resting-coupled.csv")
    accepted_rows=None
    if coupled.is_file():
        with coupled.open(newline="") as stream: accepted_rows=max(0,sum(1 for _ in csv.reader(stream))-1)
    failure_path=output/((Path() if native_dir==source else Path("native")) / "common-field-failure.json")
    failure=small_json(failure_path) if failure_path.is_file() else {}
    identities={
      "source_base_revision":provenance_data.get("source_base_revision"),
      "executable_sha256":provenance_data.get("executable_sha256",invocation_data.get("binary_sha256")),
      "metallib_sha256":provenance_data.get("metallib_sha256"),
      "suite_library_sha256":provenance_data.get("suite_library_sha256"),
      "device":provenance_data.get("device",provenance_data.get("metal_device")),
      "source_sha256":provenance_data.get("source_sha256",{}),
      "input_identity_files":raw_hashes,
      "patch_and_context":context,
    }
    (output/"raw-identities.json").write_text(json.dumps(identities,indent=2)+"\n")
    outcome={"exit_code":execution_data["exit_code"],"wall_s":execution_data.get("wall_s"),"requested_steps":int(flag_value("--muscle-step-count")) if flag_value("--muscle-step-count") else None,"step_seconds":float(flag_value("--muscle-step-seconds")) if flag_value("--muscle-step-seconds") else None,"accepted_coupled_rows":accepted_rows,"failure":failure or None,"scope":"mechanics evidence only; interpret qualification from the preregistered trial and completed run record"}
    (output/"outcome.json").write_text(json.dumps(outcome,indent=2)+"\n")
    manifest={"source_run":str(source),"copied_files":copied,"context_files":context,"omitted_large_files":omitted,"identity_record":"raw-identities.json","outcome_record":"outcome.json"}
    (output/"copy-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    sums=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name!="SHA256SUMS": sums.append(f"{sha(path)}  {path.relative_to(output).as_posix()}")
    (output/"SHA256SUMS").write_text("\n".join(sums)+"\n")
    print(json.dumps({"output":str(output),"exit_code":outcome["exit_code"],"copied_files":len(copied),"omitted_files":len(omitted),"accepted_coupled_rows":accepted_rows,"package_bytes":sum(p.stat().st_size for p in output.rglob("*") if p.is_file())},indent=2))

if __name__=="__main__": main()
