"""Thin, opt-in 535 adapter over the matched 527/523/485 loader chain."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
ENTRY_527=D/"D/D/distance-slack-scaling-523/run_distance_slack_527.py"
HELPER_535=D/"D/D/point-priority-535/point_priority_535.py"
RUNNER_485=D/"D/D/component-lp-decomposition-485/run_matched_component_fit_485.py"
RUNNER_458=D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
DECOMP_485=D/"D/D/component-lp-decomposition-485/component_lp_decomposition_485.py"
OWNER_408=D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
EXPECTED_ENTRY_527_SHA="a5d55290eab2e0673a1db59bd8c5cdc30a5f84afa4eee54bbe993943e055e245"
EXPECTED_HELPER_535_SHA="26fa5ae9114f7f65a879bcf2f33da5ceb223f1dc3a54f632742d1649c26fff16"
FLAG="--point-priority-535"
ROW_FLAG="--point-priority-query-row-535"
RETAIN_FLAG="--point-priority-retention-535"
TIME_FLAG="--point-priority-time-fraction-535"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f"cannot load {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    return mod

def _take(argv,flag,default,cast):
    if argv.count(flag)>1: raise ValueError(f"{flag} may appear once")
    if flag not in argv: return default
    i=argv.index(flag)
    if i+1>=len(argv): raise ValueError(f"{flag} needs a value")
    value=cast(argv[i+1]); del argv[i:i+2]; return value

def _instrument_458(w,helper,state,query_row,retain,time_fraction):
    if getattr(w,"_point_priority_535_wrapped",False): raise RuntimeError("535 458 installer hook already attached")
    original=w.install_objective
    def install(owner,objective,crossing):
        result=original(owner,objective,crossing)
        state["owner_hook"]=helper.patch_owner_clearance_builder(owner,state,query_row)
        return result
    w.install_objective=install
    w._point_priority_535_wrapped=True
    state["objective_options"]={"retention_fraction":retain,"priority_time_fraction":time_fraction}
    return w

def _attach(runner527,helper,state,query_row,retain,time_fraction):
    original_attach=runner527._attach_to_505_entry
    def attach(entry505,scaling,state527):
        result=original_attach(entry505,scaling,state527)
        prior_patch=entry505.patch_485_loader
        def patch_485_loader(entry490,validation,hook_states,helper505):
            prior_patch(entry490,validation,hook_states,helper505)
            prior_load=entry490._load_module
            def load_485(path):
                runner=prior_load(path)
                if Path(path).resolve()!=RUNNER_485.resolve(): return runner
                if getattr(runner,"_point_priority_535_runner_wrapped",False):
                    raise RuntimeError("535 runner hook already attached")
                prior_runner_load=runner.load
                def runner_load(name,load_path):
                    loaded=prior_runner_load(name,load_path)
                    path2=Path(load_path).resolve()
                    if path2==RUNNER_458.resolve():
                        _instrument_458(loaded,helper,state,query_row,retain,time_fraction)
                    elif path2==DECOMP_485.resolve():
                        state["decomposition_hook"]=helper.patch_component_decomposition(
                            loaded,state,retention_fraction=retain,priority_time_fraction=time_fraction)
                    return loaded
                runner.load=runner_load
                runner._point_priority_535_runner_wrapped=True
                return runner
            entry490._load_module=load_485
        entry505.patch_485_loader=patch_485_loader
        state["loader_hook"]={"path":"527 -> 505.patch_485_loader -> 490._load_module -> 485.load",
                              "attached":True}
        return result
    runner527._attach_to_505_entry=attach

def main():
    argv=list(sys.argv[1:])
    if argv.count(FLAG)!=1: raise ValueError(f"explicit opt-in {FLAG} is required exactly once")
    argv.remove(FLAG)
    query=_take(argv,ROW_FLAG,19226,int)
    retain=_take(argv,RETAIN_FLAG,.95,float)
    time_fraction=_take(argv,TIME_FLAG,.20,float)
    if query<0 or not 0<retain<=1 or not 0<time_fraction<1: raise ValueError("invalid priority target or budget")
    if not ENTRY_527.is_file() or not HELPER_535.is_file(): raise FileNotFoundError("535 loader/helper missing")
    if sha(ENTRY_527)!=EXPECTED_ENTRY_527_SHA: raise RuntimeError("535 matched 527 entry SHA mismatch")
    if sha(HELPER_535)!=EXPECTED_HELPER_535_SHA: raise RuntimeError("535 helper SHA mismatch before delegate")
    helper=load("point_priority_535_runtime",HELPER_535); state={}
    runner527=load("distance_slack_527_for_535",ENTRY_527)
    _attach(runner527,helper,state,query,retain,time_fraction)
    output=None
    original_argv=list(argv)
    if "--output" in original_argv:
        k=original_argv.index("--output")
        if k+1<len(original_argv): output=Path(original_argv[k+1])
    sys.argv[:]=[sys.argv[0],*argv]
    runner527.main()
    if sha(ENTRY_527)!=EXPECTED_ENTRY_527_SHA: raise RuntimeError("535 matched 527 entry changed during delegate")
    if sha(HELPER_535)!=EXPECTED_HELPER_535_SHA: raise RuntimeError("535 helper changed during delegate")
    if output is None or not (output/"report.json").is_file():
        print(json.dumps({"status":"535_LOADER_RETURNED_WITHOUT_FIT_REPORT","fit_or_geometry_claim":False,
                          "loader_hook":state.get("loader_hook"),"native_admission":False},sort_keys=True))
        return
    if not isinstance(state.get("capture"),dict) or not isinstance(state.get("decomposition_hook"),dict):
        raise RuntimeError("535 fit report exists but metadata/decomposition hooks did not attach")
    report_path=output/"report.json"; report=json.loads(report_path.read_text())
    report["point_priority_535"]={
        "schema":helper.SCHEMA,
        "classification":"opt-in offline construction objective pass; exact downstream gates unchanged",
        "query_row":query,"retention_fraction":retain,"priority_time_fraction":time_fraction,
        "helper_path":str(HELPER_535),"helper_sha256":sha(HELPER_535),
        "expected_entry_527_sha256":EXPECTED_ENTRY_527_SHA,"expected_helper_535_sha256":EXPECTED_HELPER_535_SHA,
        "wrapper_path":str(Path(__file__).resolve()),"wrapper_sha256":sha(__file__),
        "owner_build_capture":{k:v for k,v in state["capture"].items() if k!="priority_c_functional"},
        "owner_path":str(OWNER_408),"owner_sha256":sha(OWNER_408),
        "decomposition_hook":{k:v for k,v in state["decomposition_hook"].items() if k!="restore"},
        "loader_hook":state["loader_hook"],
        "priority_constraint":"C[metadata-selected point row] @ delta >= retention_fraction * priority LP optimum",
        "physical_hard_rows_and_output_bounds_unchanged":True,
        "one_additional_offline_construction_constraint":True,
        "native_admission":False,"accepted_physical_steps":0,
    }
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"535_PRIORITY_FIT_WRAPPED","report":str(report_path),
        "query_row":query,"point_priority_535":report["point_priority_535"],
        "native_admission":False},sort_keys=True),flush=True)

if __name__=="__main__": main()

