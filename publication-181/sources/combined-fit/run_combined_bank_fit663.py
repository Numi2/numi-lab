"""Combined objective wrapper: frozen 603/637/638 plus existing 689 weights."""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from pathlib import Path

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
ROOT=D/"D/surface-clearance-546";HERE=Path(__file__).resolve().parent
FROZEN603=ROOT/"component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
RUNNER637=ROOT/"box-aware-sat-fit-637/run_box_aware_sat637.py"
CUSTOM=("--selector-npz","--selector-report","--bank-balanced-config","--required-pairs")
PINS=HERE/"source-pins.json"
BANK_DIR=ROOT/"bank-balanced-objective-689"
AXIS_DIR=ROOT/"component-sat-selector-638"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def consume(argv):
    values={};kept=[argv[0]];i=1
    while i<len(argv):
        if argv[i] in CUSTOM:
            flag=argv[i]
            if flag in values or i+1>=len(argv):raise ValueError(f"duplicate or missing {flag}")
            values[flag]=Path(argv[i+1]).resolve();i+=2
        else:kept.append(argv[i]);i+=1
    if set(values)!=set(CUSTOM):raise ValueError("required selector/config/inventory flags are missing")
    return kept,values
def patch_source(old_patch,source):
    base=old_patch(source)
    needle="                from box_aware_sat637 import install_from_cli\n                install_from_cli(owner, objective, crossing)\n"
    insert=needle+(
      "                from sat_axis_map_adapter638 import install_axis_map_from_cli\n"
      "                install_axis_map_from_cli(owner, objective, crossing)\n"
      "                import bank_balanced_objective689 as _bank689\n"
      "                import combined_bank_coverage663 as _coverage663\n"
      "                _bank_cfg = json.loads(os.environ['NUMI_BANK_BALANCED_689_CONFIG_JSON'])\n"
      "                _required = json.loads(os.environ['NUMI_COMBINED_663_REQUIRED_PAIRS_JSON'])\n"
      "                _bank689.install(owner, objective, crossing, _bank_cfg)\n"
      "                _coverage663.install(objective, crossing, _required, _bank_cfg['candidate_sha256'], _bank_cfg['audit159_sha256'])\n")
    if base.count(needle)!=1:raise ValueError("637 install hook not unique")
    return base.replace(needle,insert,1)
def main():
    pins=load(PINS)
    for key,path in pins["sources"].items():
        if sha(path)!=pins["sha256"][key]:raise RuntimeError(f"source pin mismatch: {path}")
    fit_argv,flags=consume(sys.argv[:])
    cfg=load(flags["--bank-balanced-config"]); pair_doc=load(flags["--required-pairs"])
    ids=pair_doc.get("pair_identities")
    if pair_doc.get("schema")!="numi.human.combined_bank_target_pairs663.v1" or not isinstance(ids,list) or len(ids)!=359:
        raise ValueError("bad 359-pair inventory")
    if pair_doc.get("candidate_sha256")!=cfg.get("candidate_sha256") or pair_doc.get("audit159_sha256")!=cfg.get("audit159_sha256"):
        raise ValueError("pair inventory/config pins disagree")
    base=Path(fit_argv[fit_argv.index("--base-candidate")+1]).resolve()
    audit=Path(fit_argv[fit_argv.index("--audit159-490")+1]).resolve()
    if sha(base)!=cfg["candidate_sha256"] or sha(audit)!=cfg["audit159_sha256"]:
        raise ValueError("fit candidate/audit does not match config")
    out=Path(fit_argv[fit_argv.index("--output")+1]).resolve()
    if out.exists():raise FileExistsError(f"output must be absent: {out}")
    launch_path=out.with_name(out.name+"-combined-bank-663.launch.json")
    launch={"schema":"numi.human.combined_bank_balanced_fit663.launch.v2",
      "launcher_path":str(Path(__file__).resolve()),"launcher_sha256":sha(__file__),
      "source_pins_sha256":sha(PINS),"source_hashes":pins["sha256"],
      "selector_npz":{"path":str(flags["--selector-npz"]),"sha256":sha(flags["--selector-npz"])},
      "selector_report":{"path":str(flags["--selector-report"]),"sha256":sha(flags["--selector-report"])},
      "bank_config":{"path":str(flags["--bank-balanced-config"]),"sha256":sha(flags["--bank-balanced-config"])},
      "required_pairs":{"path":str(flags["--required-pairs"]),"sha256":sha(flags["--required-pairs"]),"count":len(ids)},
      "candidate":{"path":str(base),"sha256":sha(base)},"audit159":{"path":str(audit),"sha256":sha(audit)},
      "fit_argv_after_adapter_flags":fit_argv[1:],
      "objective_change":"Only existing 689 normalized soft-row weights are rebalanced across the three requested banks. Original face C/h/G/lower, 603 hard rows, bounds, anchors, source/trust/area/edge/self checks and final exact gates remain active.",
      "coverage_policy":"Before LP solve, verify exactly 359 requested typed pairs each have positive movable selected soft rows and exactly match the corresponding 458 guard corners; subthreshold-positive corners are reported.",
      "native_admission":False,"accepted_physical_steps":0}
    launch_path.write_text(json.dumps(launch,indent=2,sort_keys=True,allow_nan=False)+"\n")
    sys.path[:0]=[str(BANK_DIR),str(AXIS_DIR),str(ROOT/"box-aware-sat-fit-637"),str(HERE),str(FROZEN603.parent)]
    os.environ["NUMI_AXIS_MAP638_NPZ"]=str(flags["--selector-npz"])
    os.environ["NUMI_AXIS_MAP638_REPORT"]=str(flags["--selector-report"])
    os.environ["NUMI_AXIS_MAP638_ORIGINAL_FIT_ARGV_JSON"]=json.dumps(fit_argv[1:],separators=(",",":"))
    os.environ["NUMI_BANK_BALANCED_689_CONFIG_JSON"]=json.dumps(cfg,separators=(",",":"),allow_nan=False)
    os.environ["NUMI_COMBINED_663_REQUIRED_PAIRS_JSON"]=json.dumps(ids,separators=(",",":"),allow_nan=False)
    spec=importlib.util.spec_from_file_location("runner637_combined663",RUNNER637)
    if spec is None or spec.loader is None:raise ImportError(RUNNER637)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    old=runner.patch_source;runner.patch_source=lambda source:patch_source(old,source)
    oldargv=sys.argv[:];sys.argv=fit_argv;error=None
    try:runner.main()
    except BaseException as exc:error=exc
    finally:
        sys.argv=oldargv
        import bank_balanced_objective689 as bank
        import combined_bank_coverage663 as coverage
        if out.exists():
            bank.write_result(out);coverage.write_result(out)
            rp=out/"report.json"
            if rp.is_file():
                report=load(rp)
                report["combined_bank_balanced_objective_663"]={
                  "config_sha256":sha(flags["--bank-balanced-config"]),
                  "required_pair_inventory_sha256":sha(flags["--required-pairs"]),
                  "bank_install":bank.INSTALLS[-1] if bank.INSTALLS else None,
                  "soft_row_coverage":coverage.INSTALLS[-1] if coverage.INSTALLS else None,
                  "native_admission":False,"accepted_physical_steps":0}
                report["original_soft_weights_unchanged"]=False
                report["lp_equations_and_gate_sources_unchanged"]=False
                report["lp_equation_change_explanation"]="The existing 689 adapter reweighted only soft face-row weights for the three requested banks; hard rows, target vector, bounds and acceptance gates were unchanged."
                tmp=rp.with_name(rp.name+".tmp")
                tmp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
                os.replace(tmp,rp)
        else:
            launch["status"]="FIT_FAILED_BEFORE_OUTPUT";launch["error_type"]=type(error).__name__ if error else "Unknown"
            launch["error"]=str(error) if error else "unknown"
            import combined_bank_coverage663 as coverage
            launch["coverage_state"]=coverage.INSTALLS[-1] if coverage.INSTALLS else None
            launch_path.write_text(json.dumps(launch,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if error is not None:raise error.with_traceback(error.__traceback__)
if __name__=="__main__":main()

