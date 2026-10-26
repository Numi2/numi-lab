"""Direct-depth target runner with a pinned multi-parent source-normal adapter.

It delegates target construction to the frozen 545 builder. Direct-depth
direction selection and 185-bank ray/endpoint predicates remain owned by 545.
Only kind-2 multi-parent normal provenance is delegated to the tested 546
registered-source union helper; this runner is inert until explicitly invoked.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
BUILDER=ROOT/"D/contained-clearance-545/build_direct_depth_targets545.py"
BUILDER_SHA="2864d9547d2715c61c9c27b17bcfd11fac250197c47f4fd4870f8f6898cecb60"
EXT=ROOT/"D/source-outward-proposal-500/build_outward_targets_multi_source_546.py"
EXT_SHA="dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
EXT_OWNER500_SHA="454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f"cannot load {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def require_direct_depth(argv):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--direction-mode",choices=("source-outward","depth-compatible","depth-maximizing","direct-depth"),default="source-outward")
    ns,_=parser.parse_known_args(argv)
    if ns.direction_mode!="direct-depth":
        raise ValueError("this adapter requires --direction-mode direct-depth")

def install_source303_extension(owner,extension,face_provenance=None):
    """Patch only explicit kind-2 CSR unions; preserve 303 singles verbatim."""
    if face_provenance is None: face_provenance={}
    original_multi=extension.multi_parent_source_outward_normal
    def capture(face_row,candidate,detail,orientation):
        normal,prov=original_multi(face_row,candidate,detail,orientation)
        face_provenance[int(face_row)]=prov
        return normal,prov
    extension.multi_parent_source_outward_normal=capture
    original=extension.install_extension(owner)
    return original

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    require_direct_depth(argv)
    for path,expected,label in ((BUILDER,BUILDER_SHA,"545 direct-depth builder"),(EXT,EXT_SHA,"546 multi-parent helper")):
        if sha(path)!=expected: raise ValueError(f"pinned {label} hash mismatch")
    extension=loadmod(EXT,"multi_parent_normals_546_for_direct_depth")
    owner500=Path(extension.OWNER500)
    if sha(owner500)!=EXT_OWNER500_SHA or extension.OWNER500_SHA!=EXT_OWNER500_SHA:
        raise ValueError("pinned 500 source-normal owner mismatch")
    builder=loadmod(BUILDER,"direct_depth_target_builder545_for_546")
    face_provenance={}
    original_loadmod=builder.loadmod
    def load_with_multisource(path,name):
        module=original_loadmod(path,name)
        if Path(path).resolve()==Path(builder.SOURCE303).resolve():
            install_source303_extension(module,extension,face_provenance)
        return module
    builder.loadmod=load_with_multisource
    builder.main(argv)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir",type=Path,required=True)
    args,_=parser.parse_known_args(argv)
    out=args.output_dir.resolve()
    report_path=out/"nearest-union-exits.json"
    records_path=out/"nearest-union-exits.npz"
    if not report_path.is_file() or not records_path.is_file():
        raise RuntimeError("545 builder did not emit target report and arrays")
    report=json.loads(report_path.read_text())
    if report.get("direction_mode")!="direct-depth":
        raise RuntimeError("545 did not run in direct-depth mode")
    serial=json.dumps(face_provenance,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    report["normal_method"]=(
        "Area-average unique incident face normals. Single-parent faces use pinned 303 ancestry; "
        "kind-2 multi-parent faces use the 546 area-weighted registered fine-source union normal. "
        "Direction remains the matched357 direct-depth covector; 545 all185 ray/endpoint checks are unchanged."
    )
    report["multi_parent_source_normal_extension"]={
        "adapter_path":str(Path(__file__).resolve()),"adapter_sha256":sha(__file__),
        "pinned_545_builder_path":str(BUILDER),"pinned_545_builder_sha256":BUILDER_SHA,
        "pinned_546_extension_path":str(EXT),"pinned_546_extension_sha256":EXT_SHA,
        "pinned_500_owner_path":str(owner500),"pinned_500_owner_sha256":EXT_OWNER500_SHA,
        "single_parent_303_function_preserved":True,
        "multi_parent_face_rows":sorted(face_provenance),
        "multi_parent_face_count":len(face_provenance),
        "multi_parent_provenance_sha256":hashlib.sha256(serial).hexdigest(),
        "per_target_source_ancestry_recorded_by_545":True,
        "direct_depth_direction_unchanged":True,
        "exact_185_bank_ray_and_endpoint_owner_unchanged":True,
        "native_admission":False,"accepted_physical_steps":0,
    }
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(report_path),"report_sha256":sha(report_path),
        "target_npz":str(records_path),"target_npz_sha256":sha(records_path),
        "direction_mode":report["direction_mode"],"multi_parent_face_count":len(face_provenance),
        "status":report["status"]},indent=2))

if __name__=="__main__": main()
