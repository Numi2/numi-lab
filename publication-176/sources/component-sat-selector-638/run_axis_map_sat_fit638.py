"""Outer launcher that layers selector-638 routing over frozen 637/603."""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from pathlib import Path

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE=Path(__file__).resolve().parent
RUNNER637=D/"D/surface-clearance-546/box-aware-sat-fit-637/run_box_aware_sat637.py"
RUNNER637_SHA="3a142578939683981274ebef061d797b471594ce6694cd888b0e450bbaf6bff2"
HELPER637=D/"D/surface-clearance-546/box-aware-sat-fit-637/box_aware_sat637.py"
HELPER637_SHA="a226a8f7e5b4afaeeb9d484883f6a213b84220bb639a15388f3c37dd944069bd"
FROZEN603=D/"D/surface-clearance-546/component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
FROZEN603_SHA="38c3dfa9ddd961250111c87f56d6184075bde43be07c46e09d239f057b06d692"
ADAPTER=HERE/"sat_axis_map_adapter638.py"
CUSTOM=("--selector-npz","--selector-report")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def consume_flags(argv):
    vals={}
    kept=[argv[0]]
    i=1
    while i<len(argv):
        item=argv[i]
        if item in CUSTOM:
            if item in vals:raise ValueError(f"duplicate {item}")
            if i+1>=len(argv):raise ValueError(f"missing value for {item}")
            vals[item]=Path(argv[i+1]).resolve()
            i+=2
        else:
            kept.append(item);i+=1
    if set(vals)!=set(CUSTOM):raise ValueError("both --selector-npz and --selector-report are required")
    return kept,vals

def main():
    if sha(RUNNER637)!=RUNNER637_SHA or sha(HELPER637)!=HELPER637_SHA or sha(FROZEN603)!=FROZEN603_SHA:
        raise RuntimeError("frozen 603/637 source pin mismatch")
    if not ADAPTER.is_file():raise FileNotFoundError(ADAPTER)
    original_argv=sys.argv[:]
    fit_argv,selector=consume_flags(original_argv)
    out=Path(fit_argv[fit_argv.index("--output")+1]).resolve() if "--output" in fit_argv else None
    if out is None:raise ValueError("fit command must include --output")
    if out.exists():raise FileExistsError(f"fit output must be absent: {out}")
    if not out.parent.is_dir():raise FileNotFoundError(f"output parent does not exist: {out.parent}")
    launch_path=out.with_name(out.name+"-axis-map-638.launch.json")
    launch={
        "schema":"numi.human.selector_axis_map_fit638.launch.v1",
        "adapter_path":str(ADAPTER),"adapter_sha256":sha(ADAPTER),
        "launcher_path":str(Path(__file__).resolve()),"launcher_sha256":sha(__file__),
        "runner637_path":str(RUNNER637),"runner637_sha256":RUNNER637_SHA,
        "helper637_path":str(HELPER637),"helper637_sha256":HELPER637_SHA,
        "frozen603_path":str(FROZEN603),"frozen603_sha256":FROZEN603_SHA,
        "selector_npz":str(selector["--selector-npz"]),"selector_npz_sha256":sha(selector["--selector-npz"]),
        "selector_report":str(selector["--selector-report"]),"selector_report_sha256":sha(selector["--selector-report"]),
        "argv":original_argv[1:],"fit_argv_after_removing_adapter_flags":fit_argv[1:],
        "claim_limit":"Construction routing only. Selector displacement is a box witness and is not applied. Frozen 603/637 fit and all exact checks remain authoritative.",
        "native_admission":False,"accepted_physical_steps":0,
    }
    launch_path.write_text(json.dumps(launch,indent=2,sort_keys=True,allow_nan=False)+"\n")
    sys.path.insert(0,str(HERE))
    sys.path.insert(0,str(RUNNER637.parent))
    os.environ["NUMI_AXIS_MAP638_NPZ"]=str(selector["--selector-npz"])
    os.environ["NUMI_AXIS_MAP638_REPORT"]=str(selector["--selector-report"])
    os.environ["NUMI_AXIS_MAP638_ORIGINAL_FIT_ARGV_JSON"]=json.dumps(fit_argv[1:],separators=(",",":"),ensure_ascii=True)
    spec=importlib.util.spec_from_file_location("runner637_axis638",RUNNER637)
    if spec is None or spec.loader is None:raise ImportError(RUNNER637)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    old_patch=runner.patch_source
    needle="                install_from_cli(owner, objective, crossing)\n"
    injected=needle+"                from sat_axis_map_adapter638 import install_axis_map_from_cli\n                install_axis_map_from_cli(owner, objective, crossing)\n"
    def patch(source):
        base=old_patch(source)
        if base.count(needle)!=1:raise ValueError("637 post-install hook is not unique")
        return base.replace(needle,injected,1)
    runner.patch_source=patch
    sys.argv=fit_argv
    try:
        runner.main()
    finally:
        sys.argv=original_argv
        import sat_axis_map_adapter638 as adapter
        if out.exists() and adapter.INSTALLS:
            adapter.write_result(out)
        elif adapter.INSTALLS:
            fail=out.with_name(out.name+"-axis-map-638-install-failure.json")
            fail.write_text(json.dumps({"adapter":adapter.INSTALLS[-1],"fit_output_created":False,
                "native_admission":False,"accepted_physical_steps":0},indent=2,sort_keys=True,allow_nan=False)+"\n")

if __name__=="__main__":main()

