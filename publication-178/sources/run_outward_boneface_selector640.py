"""Optional wrapper for 638 that excludes only anti-outward bone-face axes.

Without --reject-opposed-bone-face this is the original selector behavior.
It remains a box-only axis-selection diagnostic; output still requires all
original LP, emitted-F32, exact self/containment and all-bank gates.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent
BASE_RUNNER=OWNER/"run_connected_patch_sat_selector638.py"
BASE_HELPER=OWNER/"connected_patch_sat_selector638.py"
FILTER_HELPER=HERE/"outward_boneface_filter640.py"
EXPECTED_BASE_RUNNER="bbaf1ca9fef90f370c5a877dd1f0249779f0091a4dad8013a581820d110ed92f"
EXPECTED_BASE_HELPER="a188e5a55d2aebb0ce27848b0d3fb03d7aa2f8715ee6a8da7bbc57d781485a9a"
EXPECTED_SAT551="27431e9ed3e92886394b59f1156a743e508fba8c81cf8b1ff94a7d80e3ef201f"
EXPECTED_FILTER_HELPER="b8ae5cc2db7e1bdfc31d668afcdc9a2113023ad2fc32c2a28622ded3bbe3ebf9"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def load_module(name,path,digest):
    if sha(path)!=digest: raise RuntimeError(f"source pin mismatch: {path}")
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def main():
    original=list(sys.argv[1:])
    enabled=original.count("--reject-opposed-bone-face")==1
    if original.count("--reject-opposed-bone-face")>1: raise SystemExit("duplicate opt-in flag")
    args=[x for x in original if x!="--reject-opposed-bone-face"]
    for path,digest in ((BASE_RUNNER,EXPECTED_BASE_RUNNER),(BASE_HELPER,EXPECTED_BASE_HELPER)):
        if sha(path)!=digest: raise RuntimeError(f"pinned 638 source changed: {path}")
    filter_mod=load_module("outward_boneface_filter640",FILTER_HELPER,EXPECTED_FILTER_HELPER)
    sys.path.insert(0,str(OWNER))
    core=load_module("sat_selector_core640",BASE_HELPER,EXPECTED_BASE_HELPER)
    runner=load_module("sat_selector_runner640",BASE_RUNNER,EXPECTED_BASE_RUNNER)
    runner.selector=core
    original_loader=runner.load_module
    receipts=[]
    def patched_loader(name,path,digest):
        module=original_loader(name,path,digest)
        if name=="sat551_for_selector638":
            if sha(path)!=EXPECTED_SAT551: raise RuntimeError("pinned 551 axis source changed")
            base_axes=module._signed_sat_axes
            def filtered_axes(skin,bone):
                axes=base_axes(skin,bone)
                kept,receipt=filter_mod.filter_axis_inventory(axes,bone,enabled=enabled)
                receipts.append(receipt)
                return kept
            module._signed_sat_axes=filtered_axes
        return module
    runner.load_module=patched_loader
    output=None
    for i,arg in enumerate(args[:-1]):
        if arg=="--output": output=Path(args[i+1]).resolve();break
    if output is None: raise SystemExit("--output is required")
    sys.argv=[sys.argv[0],*args]
    runner.main()
    # Add the opt-in policy receipt to whichever frozen selector artifact was
    # emitted, without changing the baseline 638 implementation or NPZ schema.
    report_path=output/"preflight.json"
    if not report_path.exists(): report_path=output/"report.json"
    if report_path.exists():
        report=json.loads(report_path.read_text())
        report["outward_boneface_filter640"]={
            "enabled":enabled,
            "implementation_sha256":sha(FILTER_HELPER),
            "wrapper_sha256":sha(Path(__file__)),
            "base_selector_runner_sha256":sha(BASE_RUNNER),
            "base_selector_helper_sha256":sha(BASE_HELPER),
            "sat_axis_provider_sha256":EXPECTED_SAT551,
            "per_pair_calls":len(receipts),
            "input_axis_count":sum(int(r["input_axis_count"]) for r in receipts),
            "output_axis_count":sum(int(r["output_axis_count"]) for r in receipts),
            "removed_axis_count":sum(int(r["removed_axis_count"]) for r in receipts),
            "pairs_with_removed_axis":sum(bool(r["removed_axis_count"]) for r in receipts),
            "rule":"when enabled, remove only a bone_face axis exactly opposing the ordered fixed-bone triangle normal; retain skin_face and edge_cross axes",
            "limitations":["box-only selector output is not clearance", "the existing fit and all exact construction gates remain mandatory", "bone triangle winding must be consistently outward"]}
        report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")

if __name__=="__main__":
    main()
