from __future__ import annotations
import hashlib,json,subprocess,time,os
from pathlib import Path
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/outward-boneface-filter-640")
SOURCE=ROOT.parent/"fit-from-675-001/actual-fit-argv.json"
SELECTOR=ROOT/"selector-675-reject-opposed-001"
OUT=ROOT/"fit-675-reject-opposed-001"
EXPECTED_SOURCE="e224f8827eb2cfd713fa84c63dfb6ac1f974855c36b8098774d0902409a2257d"
EXPECTED_SELECTOR_NPZ="bc850c2db280c06c060136987c58cce8fea5e3c2577fed2723771dca2168dfb0"
EXPECTED_SELECTOR_REPORT="8ee66a781ebf64dc76a7dd52e3725e935955d1c1da8d3dccf691c9b89261e5ad"
FIT_SCRIPT=ROOT.parent/"run_axis_map_sat_fit638.py"
FIT_SHA="42dc5c6b6447e0bac2256cd746e6247a66aee2b1555d498521a4c9df9faed649"
FILTER_RUNNER=ROOT/"run_outward_boneface_selector640.py"
FILTER_HELPER=ROOT/"outward_boneface_filter640.py"
BASE_SELECTOR=ROOT.parent/"run_connected_patch_sat_selector638.py"
BASE_HELPER=ROOT.parent/"connected_patch_sat_selector638.py"
SAT551=ROOT.parent.parent/"pair-sat-routing-551/pair_sat_routing_551.py"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def set_value(argv,flag,value):
 if argv.count(flag)!=1:raise ValueError(f"expected one {flag}")
 i=argv.index(flag)
 if i+1>=len(argv):raise ValueError(f"missing {flag} value")
 argv[i+1]=str(value)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 for p,d in [(SOURCE,EXPECTED_SOURCE),(SELECTOR/"selection.npz",EXPECTED_SELECTOR_NPZ),(SELECTOR/"report.json",EXPECTED_SELECTOR_REPORT)]:
  if sha(p)!=d:raise RuntimeError(f"input pin mismatch: {p}")
 argv=json.loads(SOURCE.read_text())
 if not isinstance(argv,list) or argv[1]!=str(FIT_SCRIPT):raise ValueError("fit argv does not target pinned 638 launcher")
 cmd=list(argv)
 set_value(cmd,"--selector-npz",SELECTOR/"selection.npz")
 set_value(cmd,"--selector-report",SELECTOR/"report.json")
 set_value(cmd,"--output",OUT/"fit")
 expected=[(FIT_SCRIPT,FIT_SHA),(FILTER_RUNNER,"378a6fd8f0b00e2be2dfa212205004e1362dc4002b2088b1c33a2bb608bcd768"),(FILTER_HELPER,"b8ae5cc2db7e1bdfc31d668afcdc9a2113023ad2fc32c2a28622ded3bbe3ebf9"),(BASE_SELECTOR,"bbaf1ca9fef90f370c5a877dd1f0249779f0091a4dad8013a581820d110ed92f"),(BASE_HELPER,"a188e5a55d2aebb0ce27848b0d3fb03d7aa2f8715ee6a8da7bbc57d781485a9a"),(SAT551,"27431e9ed3e92886394b59f1156a743e508fba8c81cf8b1ff94a7d80e3ef201f")]
 pins={str(p):sha(p) for p,_ in expected}
 for p,d in expected:
  if d and pins[str(p)]!=d:raise RuntimeError(f"source pin mismatch: {p}")
 OUT.mkdir(parents=True)
 launch={"schema":"numi.outward_boneface_fit640.launch.v1","argv":cmd,"original_fit_argv_path":str(SOURCE),"original_fit_argv_sha256":sha(SOURCE),"selector_npz":str(SELECTOR/"selection.npz"),"selector_npz_sha256":sha(SELECTOR/"selection.npz"),"selector_report":str(SELECTOR/"report.json"),"selector_report_sha256":sha(SELECTOR/"report.json"),"source_pins":pins,"selection_mode":"640 opt-in rejects only anti-outward bone_face signed axis; all other SAT axes retained","claim_limit":"existing 603/637 fit only; no full exact self, containment, all-bank, or physical admission is performed by this launcher","native_admission":False,"accepted_physical_steps":0}
 (OUT/"launch.json").write_text(json.dumps(launch,indent=2,sort_keys=True)+"\n")
 log=OUT/"fit-stage.log";started=time.time()
 with log.open("wb") as f: result=subprocess.run(cmd,cwd="/Users/numi/Work/numi-human-connected-surface-069",stdout=f,stderr=subprocess.STDOUT,check=False)
 execution={"schema":"numi.outward_boneface_fit640.execution.v1","returncode":int(result.returncode),"elapsed_seconds":time.time()-started,"log":str(log),"log_sha256":sha(log),"fit_output":str(OUT/"fit"),"source_pins_after":{str(p):sha(p) for p,_ in expected},"native_admission":False,"accepted_physical_steps":0}
 (OUT/"execution.json").write_text(json.dumps(execution,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":"DONE" if result.returncode==0 else "FAILED","returncode":result.returncode,"output":str(OUT),"elapsed_seconds":execution["elapsed_seconds"]},sort_keys=True))
if __name__=="__main__":main()
