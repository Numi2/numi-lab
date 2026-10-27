"""Opt-in wrapper around unchanged 638 -> 637 -> 603; captures one LP build."""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from pathlib import Path

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE=Path(__file__).resolve().parent
ROOT= D/"D/surface-clearance-546"
LAUNCH638=ROOT/"component-sat-selector-638/run_axis_map_sat_fit638.py"
ADAPTER638=ROOT/"component-sat-selector-638/sat_axis_map_adapter638.py"
CAPTURE=HERE/"capture_lp_system639.py"
SOURCES={
str(LAUNCH638):"42dc5c6b6447e0bac2256cd746e6247a66aee2b1555d498521a4c9df9faed649",
str(ADAPTER638):"d49c33a0f3108e1005b6fc64357f9edc796a0c469d9ba61ac7378915939365f6",
str(ROOT/"box-aware-sat-fit-637/run_box_aware_sat637.py"):"3a142578939683981274ebef061d797b471594ce6694cd888b0e450bbaf6bff2",
str(ROOT/"box-aware-sat-fit-637/box_aware_sat637.py"):"a226a8f7e5b4afaeeb9d484883f6a213b84220bb639a15388f3c37dd944069bd",
str(ROOT/"component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"):"38c3dfa9ddd961250111c87f56d6184075bde43be07c46e09d239f057b06d692",
str(D/"D/D/component-lp-decomposition-485/component_lp_decomposition_485.py"):"b8632e0734ff0163e854b9c0c5fbfe33c6d12c7b17cbb7935c5028c8df78ef74",
str(D/"D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"):"7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693",
str(ROOT/"component-scp-area-backtrack-603/frozen-v1/component_lp_partial_timeout564.py"):"a24aa58178394e25d7b38c7c35a36c93eaf109a666ace7773c0ca13d33250818",
str(ROOT/"component-scp-area-backtrack-603/frozen-v1/component_scp_backtracking603.py"):"f02aed5e462779e85a5d4df46a38a68bcf7eb3972c9dbf565830c4919b387be6",
str(ROOT/"raw-joint-capture-563/raw_joint_capture563.py"):"fa7df8771a37200c4ed30e61d92ebf1b6227eed9da8a5ff45ab54b4ff299975e",
str(ROOT/"joint-objective-full-coordinate-560/joint_objective_560.py"):"09ce0c4f47e69142de6ce394576edd3ef0ee6f4b946824a476f5425919c492c8",
str(ROOT/"surface_objective_546.py"):"dd6f70afab9cb51f3658d8f4fc9d65a13a6f70517e536a9e81c4c2a098750516",
str(D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"):"5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2",
str(D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"):"08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93",
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def arg_value(argv,flag):
 if argv.count(flag)!=1:raise ValueError(f"expected one {flag}")
 i=argv.index(flag)
 if i+1>=len(argv):raise ValueError(f"missing {flag}")
 return Path(argv[i+1]).resolve()
def _load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise ImportError(path)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
 for p,digest in SOURCES.items():
  if sha(p)!=digest:raise RuntimeError(f"pinned source changed: {p}")
 argv=sys.argv[:]
 if len(argv)<2:raise ValueError("fit arguments required")
 out=arg_value(argv,"--output")
 base=arg_value(argv,"--base-candidate")
 selector=arg_value(argv,"--selector-npz")
 selector_report=arg_value(argv,"--selector-report")
 if out.exists():raise FileExistsError(f"fit output must be absent: {out}")
 if not out.parent.is_dir():raise FileNotFoundError(out.parent)
 capdir=out.with_name(out.name+"-lp-system-639")
 if capdir.exists():raise FileExistsError(capdir)
 launch_path=out.with_name(out.name+"-lp-system-639.launch.json")
 if launch_path.exists():raise FileExistsError(launch_path)
 helper=_load("capture_lp_system639",CAPTURE)
 own_pins={"launcher_sha256":sha(__file__),"capture_helper_sha256":sha(CAPTURE)}
 adapter=_load("sat_axis_map_adapter638",ADAPTER638)
 capture=helper.Capture(capdir,source_pins={p:d for p,d in SOURCES.items()},
     argv=argv[1:],base_candidate=base,selector_npz=selector,selector_report=selector_report,max_calls=1)
 original_install=adapter.install_axis_map_from_cli
 adapter.install_axis_map_from_cli=helper.install_capture_after_axis_adapter(original_install,capture)
 launch={"schema":"numi.human.hard_lp_system_capture639.launch.v1",
  "launcher_path":str(Path(__file__).resolve()),"launcher_sha256":sha(__file__),
  "capture_helper_path":str(CAPTURE),"capture_helper_sha256":own_pins["capture_helper_sha256"],
  "source_pins":SOURCES,"argv":argv[1:],"base_candidate_sha256":sha(base),
  "selector_npz_sha256":sha(selector),"selector_report_sha256":sha(selector_report),
  "fit_output":str(out),"capture_output":str(capdir),
  "claim":"One exact first-call capture of existing hard rows A/b, soft rows C/h/weights, output bounds, target and basis map. All frozen owners and acceptance gates remain authoritative.",
  "construction_only":True,"native_admission":False,"accepted_physical_steps":0}
 launch_path.write_text(json.dumps(launch,indent=2,sort_keys=True,allow_nan=False)+"\n")
 runner=_load("run_axis_map_sat_fit638_for_capture639",LAUNCH638)
 status="returned";error=None
 try:
  runner.main()
 except BaseException as exc:
  status="raised";error=exc
  raise
 finally:
  capture.finalize(status,error)
  checks={p:{"expected":digest,"actual":(sha(p) if Path(p).is_file() else None)}
          for p,digest in SOURCES.items()}
  checks["launcher"]={"expected":own_pins["launcher_sha256"],"actual":sha(__file__)}
  checks["capture_helper"]={"expected":own_pins["capture_helper_sha256"],"actual":sha(CAPTURE)}
  ok=all(v["expected"]==v["actual"] for v in checks.values())
  receipt={"schema":"numi.human.hard_lp_system_capture639.source_integrity.v1",
           "all_pins_unchanged":ok,"checks":checks,
           "native_admission":False,"accepted_physical_steps":0}
  if capdir.exists():
   (capdir/"source-integrity.json").write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
  else:
   launch_path.with_name(launch_path.stem+"-source-integrity.json").write_text(
       json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
  if not ok and error is None:
   raise RuntimeError("capture source pin changed during fit invocation")
if __name__=="__main__":main()
