#!/usr/bin/env python3
"""Prepare publication 054 on a separate Git index; never updates refs or pushes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path("/Users/n/numi-human-aortic-contact-opt-20261019").resolve()
EVIDENCE = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
PATCH012 = EVIDENCE / "global-skin-crossing-patches-012"
TRANSFER013 = EVIDENCE / "E/connected-skin-mechanical-transfer-013"
OUTDIR = EVIDENCE / "publication-054"
PARENT = "62b59e81d8fa3de8e8fadd44035f02704f3fe763"
MANIFEST_REL = str((OUTDIR / "publication-set-054.json").relative_to(REPO))
FILELIST_REL = str((OUTDIR / "publication-filelist-054.tsv").relative_to(REPO))
MANIFEST_PATH = REPO / MANIFEST_REL
FILELIST_PATH = REPO / FILELIST_REL
PREPARED_DIR = Path("/tmp")

def git(*args: str, env: dict[str, str] | None = None, input_bytes: bytes | None = None) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO), *args], env=env, input=input_bytes)

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def git_blob_sha1(path: Path) -> str:
    h=hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def parent_tree() -> dict[str, tuple[str,str]]:
    out={}
    for entry in git("ls-tree","-r","-z",PARENT).split(b"\0"):
        if not entry: continue
        meta,raw_path=entry.split(b"\t",1)
        mode,kind,oid=meta.decode("ascii").split()
        if kind=="blob": out[raw_path.decode("utf-8")]=(mode,oid)
    return out

def selected_sources() -> tuple[list[Path],list[dict[str,str]]]:
    files:set[Path]=set()
    external=[]
    def add(path:Path)->None:
        if path.is_symlink(): raise RuntimeError(f"refusing symlink source: {path}")
        p=path.resolve(strict=True)
        if not p.is_file(): raise RuntimeError(f"not a regular file: {p}")
        if not p.is_relative_to(REPO): raise RuntimeError(f"source escapes repository: {p}")
        files.add(p)
    def add_tree(root:Path)->None:
        root=root.resolve(strict=True)
        for p in sorted(root.rglob("*")):
            if any(part=="__pycache__" or part.endswith(".pyc") for part in p.relative_to(root).parts): continue
            if p.is_file() and not p.is_symlink(): add(p)
            elif p.is_symlink(): raise RuntimeError(f"symlink in selected evidence tree: {p}")
    # Complete 012 extraction package, tests, reports, bounds, and explicit unresolved-map record.
    add_tree(PATCH012)
    # 013 source, raw predecessor and corrected candidate, reports, and test sources.
    for rel in (
        "ConnectedSkinMechanicalTransfer013.hpp",
        "build_boundary_refinement_013.py",
        "build_boundary_refinement_013_v2.py",
        "test_ConnectedSkinMechanicalTransfer013.cpp",
        "test_boundary_refinement_013.py",
        "boundary-refinement-001/candidate.npz",
        "boundary-refinement-001/report.json",
        "boundary-refinement-001/run.log",
        "boundary-refinement-002/candidate.npz",
        "boundary-refinement-002/freeze-manifest.json",
        "boundary-refinement-002/report.json",
        "boundary-refinement-002/run.log",
    ):
        add(TRANSFER013/rel)
    # The publication's narrowly rerun regression receipts and this script itself.
    for rel in ("python-tests.log","cpp-tests.log","test-results-054.json","prepare_publication_054.py"):
        add(OUTDIR/rel)
    # Exact upstream inputs used by builder 013. The old 014 file is only an
    # immutable preflight dependency; later root014/016/017 and active015 experiments
    # are deliberately excluded.
    deps=[
      EVIDENCE/"global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz",
      PATCH012/"connected-physical-O-patches-012.npz",
      EVIDENCE/"global-inner-p1-edge-support-403/refine_internal_o_edges_403.py",
      EVIDENCE/"global-connected-inner-field-293-cavity-remesh-014-independent-volume-001/field-transfer-preflight.npz",
      EVIDENCE/"bilateral-source-cut-plane-271/candidate-271-max-cut-distance-bilateral-collar.npz",
    ]
    for p in deps: add(p)
    # Add every 012 pinned input that belongs to this repository; record the
    # separate source-294 payload as an external hash-pinned dependency.
    source_manifest=json.loads((PATCH012/"source-output-manifest-012.json").read_text())
    for raw_path,expected in source_manifest["sha256"].items():
        p=Path(raw_path)
        if p.is_relative_to(REPO):
            add(p)
            if sha256_file(p)!=expected: raise RuntimeError(f"012 pinned input changed: {p}")
        else:
            if not p.is_file() or sha256_file(p)!=expected: raise RuntimeError(f"012 external input unavailable/changed: {p}")
            external.append({"path":str(p),"sha256":expected,"bytes":p.stat().st_size,"reason":"externally stored detailed-294 payload; pinned by 012, not copied into this repository"})
    return sorted(files,key=lambda p:str(p.relative_to(REPO))),external

def file_rows(tree:dict[str,tuple[str,str]])->list[dict]:
    sources,external=selected_sources()
    rows=[]
    for p in sources:
        rel=str(p.relative_to(REPO)); oid=git_blob_sha1(p); inherited=tree.get(rel)
        state="new" if inherited is None else ("unchanged_in_parent" if inherited[1]==oid else "updated_in_parent_path")
        rows.append({"path":rel,"source_path":str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p),"git_blob_sha1":oid,"parent_state":state,"stage":state!="unchanged_in_parent"})
    return rows,external

def verify_013()->dict:
    report=json.loads((TRANSFER013/"boundary-refinement-002/report.json").read_text())
    freeze=json.loads((TRANSFER013/"boundary-refinement-002/freeze-manifest.json").read_text())
    candidate=TRANSFER013/"boundary-refinement-002/candidate.npz"
    if sha256_file(candidate)!=report["output_sha256"]: raise RuntimeError("013 corrected candidate/report hash mismatch")
    if sha256_file(TRANSFER013/"build_boundary_refinement_013_v2.py")!=freeze["files"]["builder"]["sha256"]: raise RuntimeError("013 builder/freeze hash mismatch")
    if sha256_file(candidate)!=freeze["files"]["candidate"]["sha256"]: raise RuntimeError("013 candidate/freeze hash mismatch")
    if report["ancestry"]["new_detailed_294_mapping_supplied"] is not False: raise RuntimeError("unexpected 013 detailed-294 ancestry status")
    if report["quality"]["nonpositive_F32_determinants"]!=0 or report["quality"]["nonpositive_F64_determinants"]!=0: raise RuntimeError("013 candidate contains nonpositive cell determinants")
    return report

def make_manifest()->dict:
    remote=git("ls-remote","origin","refs/heads/main").decode().split()[0]
    if remote!=PARENT: raise RuntimeError(f"origin/main changed: expected {PARENT}, got {remote}")
    tree=parent_tree()
    rows,external= file_rows(tree)
    if any("/root014" in x["path"] or "/root016" in x["path"] or "/root017" in x["path"] or "/active015" in x["path"] for x in rows):
        raise RuntimeError("excluded experimental candidate entered publication selection")
    report013=verify_013()
    mapping=json.loads((PATCH012/"source294-mapping-status-012.json").read_text())
    if mapping["status"]!="PROVISIONAL_MAP_FAILURE_UNRESOLVED": raise RuntimeError("012 unresolved-map record missing")
    filelist="".join(f'{x["sha256"]}\t{x["bytes"]}\t{x["path"]}\n' for x in rows)
    FILELIST_PATH.write_text("# publication 054 source files; manifest and this list are excluded to avoid self-reference\n"+filelist)
    test_receipts=json.loads((OUTDIR/"test-results-054.json").read_text())
    if test_receipts["python"]["status"]!="PASS" or test_receipts["cpp"]["status"]!="PASS": raise RuntimeError("publication regression receipts are not passing")
    return {
      "schema":"numi.human.publication-set-054.v1",
      "purpose":"Prepare a scoped publication on an alternate Git index. This script never pushes or updates refs.",
      "parent_commit":PARENT,
      "origin_main_at_manifest_generation":remote,
      "normal_checkout_head_at_manifest_generation":git("rev-parse","HEAD").decode().strip(),
      "publication_admission":{"conforming_boundary_refinement":True,"source294_mapping":False,"mechanical":False,"physical":False,"native":False,"performance":False},
      "scope":{
        "012":{"status":"PASS_EXTRACTION_AND_SOURCE_CSR_LINEAGE; SOURCE294_MAP_UNCERTIFIED","patches":9,"seed_faces":426,"selected_O_faces":1171,"selected_O_nodes":808,"canonical_exact_pairs":11251,"exact_hit_points":22502,"max_hit_reconstruction_residual_m":1.5947552845505132e-14,"source294_mapping_status":mapping["status"],"source294_unmatched_rows":mapping["unmatched_rows"]},
        "013":{"status":report013["status"],"midpoint_edges":report013["output_counts"]["nodes"]-report013["input_counts"]["nodes"],"output_shell_triangles":report013["output_counts"]["shell_faces"],"min_normalized_det_F64":report013["quality"]["min_normalized_det_F64"],"nonpositive_F32_determinants":report013["quality"]["nonpositive_F32_determinants"],"transfer_005_max_abs_column_error":report013["volume"]["max_transfer_column_closure_after_conservative_within_parent_redistribution"]["005"]["max_abs_column_sum_error"],"transfer_014_max_abs_column_error":report013["volume"]["max_transfer_column_closure_after_conservative_within_parent_redistribution"]["014"]["max_abs_column_sum_error"],"detailed294_mapping":"not emitted"}
      },
      "limitations":[
        "013 is a conforming O/FEM boundary-refinement candidate only; geometry coordinates in the old prefix are bitwise unchanged and 857 midpoint nodes are appended.",
        "013 has no verified detailed-294 barycentric embedding; its 014/remesh ancestry is not a substitute for typed detail transfer.",
        "012 source-294 nearest map remains a recorded provisional failure: 85 of 808 O nodes are unmatched and remaining nearest candidates are not certified ancestry.",
        "No full-bank collision clearance, complete mechanics/soft-tissue acceptance, physical admission, native FEM recompilation/run, or performance claim.",
        "boundary-refinement-001 and build_boundary_refinement_013.py are retained as the raw superseded attempt; corrected 002 is the tested source, not a promotion of 001."
      ],
      "tests":test_receipts,
      "files":rows,
      "filelist":{"path":FILELIST_REL,"sha256":sha256_file(FILELIST_PATH),"bytes":FILELIST_PATH.stat().st_size,"rows":len(rows)},
      "external_inputs":external,
      "excluded_experiments":["root014","root016","root017","active015"],
      "totals":{"selected_source_files":len(rows),"staged_files_excluding_manifest_and_filelist":sum(r["stage"] for r in rows),"selected_bytes":sum(r["bytes"] for r in rows),"staged_bytes_excluding_manifest_and_filelist":sum(r["bytes"] for r in rows if r["stage"]),"unchanged_parent_files":sum(r["parent_state"]=="unchanged_in_parent" for r in rows)}
    }

def check_state()->tuple[Path,str,str|None]:
    index=Path(git("rev-parse","--path-format=absolute","--git-path","index").decode().strip())
    head=git("rev-parse","HEAD").decode().strip()
    index_sha=sha256_file(index) if index.exists() else None
    if index.with_name(index.name+".lock").exists(): raise RuntimeError(f"normal Git index locked: {index}.lock")
    return index,head,index_sha

def prepare_commit()->dict:
    manifest=json.loads(MANIFEST_PATH.read_text())
    if manifest.get("schema")!="numi.human.publication-set-054.v1" or manifest.get("parent_commit")!=PARENT: raise RuntimeError("manifest schema/parent mismatch")
    if git("ls-remote","origin","refs/heads/main").decode().split()[0]!=PARENT: raise RuntimeError("origin/main no longer equals requested parent")
    tree_parent=parent_tree()
    for row in manifest["files"]:
        p=REPO/row["path"]
        if p.is_symlink(): raise RuntimeError(f"refusing source symlink: {p}")
        p=p.resolve(strict=True)
        if not p.is_file() or not p.is_relative_to(REPO): raise RuntimeError(f"unsafe source: {p}")
        if p.stat().st_size!=row["bytes"] or sha256_file(p)!=row["sha256"] or git_blob_sha1(p)!=row["git_blob_sha1"]: raise RuntimeError(f"source pin mismatch: {p}")
        old=tree_parent.get(row["path"])
        current="new" if old is None else ("unchanged_in_parent" if old[1]==row["git_blob_sha1"] else "updated_in_parent_path")
        if current!=row["parent_state"]: raise RuntimeError(f"parent state changed: {row['path']}")
    if sha256_file(FILELIST_PATH)!=manifest["filelist"]["sha256"]: raise RuntimeError("filelist hash mismatch")
    index,head_before,index_sha_before=check_state()
    env=dict(os.environ,GIT_INDEX_FILE=str(PREPARED_DIR/f"publication-054.{os.getpid()}.index"))
    alt=Path(env["GIT_INDEX_FILE"])
    if alt.exists(): raise RuntimeError(f"refusing to overwrite existing alternate index: {alt}")
    git("read-tree",PARENT,env=env)
    paths=[r["path"] for r in manifest["files"] if r["stage"]]
    paths.extend([FILELIST_REL,MANIFEST_REL])
    for i in range(0,len(paths),150): git("add","-f","--",*paths[i:i+150],env=env)
    tree=git("write-tree",env=env).decode().strip()
    diff=subprocess.run(["git","-C",str(REPO),"-c","core.whitespace=-blank-at-eof","diff","--check",PARENT,tree],capture_output=True,text=True)
    if diff.returncode: raise RuntimeError("git diff --check failed: "+diff.stdout+diff.stderr)
    changed=git("diff-tree","--no-commit-id","--name-status","-r",PARENT,tree).decode()
    bad=[line for line in changed.splitlines() if not line.endswith(tuple(r["path"] for r in manifest["files"] if r["stage"])) and not line.endswith((FILELIST_REL,MANIFEST_REL))]
    if bad: raise RuntimeError("alternate tree contains unselected paths: "+repr(bad[:10]))
    msg=("feat(human): publish physical-O patch inputs and conforming transfer refinement 054\n\n"
         "Add the validated 012 physical-O crossing patch extraction and the tested 013 conforming boundary-refinement candidate, retaining raw and corrected attempts with hash-pinned provenance. The 012 source-294 nearest mapping remains explicitly unresolved; the 013 detailed-surface embedding is not emitted.\n\n"
         "No full-bank mechanical, physical, native, or performance admission is claimed.\n")
    commit=subprocess.run(["git","-C",str(REPO),"commit-tree",tree,"-p",PARENT],input=msg,text=True,capture_output=True,check=True).stdout.strip()
    index_after,head_after,index_sha_after=check_state()
    remote_after=git("ls-remote","origin","refs/heads/main").decode().split()[0]
    if head_after!=head_before or index_sha_after!=index_sha_before or remote_after!=PARENT: raise RuntimeError("normal HEAD/index or remote parent changed")
    return {"prepared_commit_object":commit,"tree":tree,"parent":PARENT,"origin_main_checked":remote_after,"normal_head_unchanged":head_before,"normal_index_path":str(index_after),"normal_index_sha256_unchanged":index_sha_before,"alternate_index":str(alt),"manifest_path":str(MANIFEST_PATH),"manifest_sha256":sha256_file(MANIFEST_PATH),"filelist_path":str(FILELIST_PATH),"filelist_sha256":sha256_file(FILELIST_PATH),"changed_paths":changed,"diff_check":"PASS (only blank-at-eof ignored)","pushed":False,"refs_updated":False,"physical_admission":False,"native_admission":False}

def test_receipts()->dict:
    return json.loads((OUTDIR/"test-results-054.json").read_text())

def main():
    parser=argparse.ArgumentParser()
    g=parser.add_mutually_exclusive_group()
    g.add_argument("--manifest-only",action="store_true")
    g.add_argument("--prepare",action="store_true")
    parser.add_argument("--refresh-manifest",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_commit(),indent=2))
    else:
        if MANIFEST_PATH.exists() and not args.refresh_manifest: raise RuntimeError(f"refusing to overwrite {MANIFEST_PATH}")
        result=make_manifest()
        MANIFEST_PATH.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"manifest":str(MANIFEST_PATH),"sha256":sha256_file(MANIFEST_PATH),"filelist":str(FILELIST_PATH),"filelist_sha256":sha256_file(FILELIST_PATH),"selected":len(result["files"]),"staged":result["totals"]["staged_files_excluding_manifest_and_filelist"]},indent=2))
if __name__=="__main__": main()
