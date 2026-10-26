#!/usr/bin/env python3
"""Run pinned prep-417 with the hash-pinned, ancestry-validating 335 copy."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
PREP417=ROOT/"D/prepare_connected_continuation_inputs_417.py"
ORIGINAL335=ROOT/"barycentric-connected-clearance-170/source-outward-proposal-300/build_outward_nearest_union_targets_335.py"
OWNER500=ROOT/"D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py"
OWNER390=ROOT/"D/coherent-point-targets-390/build_coherent_point_targets_390.py"
OWNER026=ROOT/"barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"
OWNER357=ROOT/"D/point-depth-barriers-357/point_depth_barriers_357.py"
PINS={
  "prep417":"c873b6d39590fe1d1baeafc1c65109b3179c292baa747f676f325de10357e7a8",
  "original335":"1c3468b6e6285afed77e8ae7ca463abb797f6a22d562ccf878734c39161947c9",
  "owner500":"454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf",
  "owner390":"15be65c3f3863998a2fb709e70e22a20402ab1580e84960aeb1a3f2301c70b00",
  "owner026":"c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3",
  "owner357":"dc8078515dd12b2edc62c0bc75759a7cb9472c9914e4704ecd41e04c4d5378e1",
}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_paths():
  return {"prep417":PREP417,"original335":ORIGINAL335,"owner500":OWNER500,
          "owner390":OWNER390,"owner026":OWNER026,"owner357":OWNER357}

def verify_source_pins():
  actual={k:sha(v) for k,v in source_paths().items()}
  mismatch={k:{"expected":PINS[k],"actual":actual[k]} for k in PINS if actual[k]!=PINS[k]}
  if mismatch: raise ValueError(f"501 source pin mismatch: {mismatch}")
  return actual

def load_pinned_preparer():
  verify_source_pins()
  spec=importlib.util.spec_from_file_location("pinned_prepare_417_with_500",PREP417)
  if spec is None or spec.loader is None: raise RuntimeError("cannot load pinned 417 preparer")
  prep=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prep;spec.loader.exec_module(prep)
  if Path(prep.BUILDER335).resolve()!=ORIGINAL335.resolve(): raise ValueError("417 default 335 path changed")
  if prep.PINS.get("builder335")!=PINS["original335"]: raise ValueError("417 original 335 pin changed")
  prep.BUILDER335=OWNER500
  prep.PINS=dict(prep.PINS)
  prep.PINS["builder335"]=PINS["owner500"]
  return prep

def output_root_from_argv(argv):
  for i,arg in enumerate(argv[:-1]):
    if arg=="--output-root": return Path(argv[i+1])
  raise ValueError("--output-root is required")

def main():
  before=verify_source_pins()
  prep=load_pinned_preparer()
  out=output_root_from_argv(sys.argv[1:])
  failure=None
  try:
    prep.main()
  except BaseException as exc:
    failure=f"{type(exc).__name__}: {exc}"
    raise
  finally:
    after={k:sha(v) for k,v in source_paths().items()}
    unchanged=after==before
    receipt={
      "schema":"numi.human.prep417-surface-owner-override-501.v1",
      "classification":"construction-target-retry-only; full-185 and exact crossing-count checks remain in pinned 417",
      "prep417_path":str(PREP417),"original_335_path":str(ORIGINAL335),"replacement_335_path":str(OWNER500),
      "pins":PINS,"source_pins_before":before,"source_pins_after":after,
      "source_pins_unchanged":unchanged,"failure":failure,
      "all_185_validation_bypassed":False,"accepted_physical_steps":0,"native_admission":False,
    }
    if out.is_dir():
      (out/"surface-owner-override-501.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    if not unchanged: raise RuntimeError("501 source changed during prep run; see override receipt")

if __name__=="__main__": main()
