"""Run the unchanged 597 certification on successfully completed target output."""
from pathlib import Path
import argparse,importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
p=N/"source-current-targets-597/verify_recovered_targets597.py"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--folder",type=Path,required=True);ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 sp=importlib.util.spec_from_file_location("verify597_for604",p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 r=m.verify(a.folder,a.state)
 r["classification"]="unchanged 597 independent certification of completed construction targets; no geometry or physical admission"
 r["source_pins"][str(Path(__file__).resolve())]=m.sha(__file__)
 a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:r[k]for k in ("status","target_count","joint_source_current_targets","source_only_exception_targets")}))
if __name__=="__main__":main()
