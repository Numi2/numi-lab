"""Refresh current witnesses using the tested kind-0 source ancestry owner."""
from pathlib import Path
import hashlib, importlib.util, sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=D/"D/refresh-491/run_refresh491.py"
ADAPTER=D/"D/source-outward-proposal-500/run_prepare_417_with_surface_owner_500.py"
SOURCE=ADAPTER.with_name("build_outward_nearest_union_targets_500.py")
PINS={BASE:"558d220e89eb2f13b24be068bebcb6f59d43b4e7934448b85b67ee89b746137b",ADAPTER:"726a0e9ae7de490f097cf2e1c02114a0042cb7b4509a2b2e726089b6a991a4d8",SOURCE:"454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(base):
 original=base.owner_command
 def command(template,owner):
  argv=original(template,owner)
  if Path(owner).resolve()==base.PREP.resolve():
   if argv[1]!=str(base.PREP):raise ValueError("unexpected preparation owner argv")
   argv[1]=str(ADAPTER)
  return argv
 base.owner_command=command
 base.EXTRA_SOURCES=[*base.EXTRA_SOURCES,ADAPTER,SOURCE,Path(__file__)]
def main():
 for p,h in PINS.items():
  if sha(p)!=h:raise ValueError(f"510 frozen source mismatch: {p}")
 spec=importlib.util.spec_from_file_location("frozen_refresh491_for510",BASE)
 base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
 install(base);base.main()
 for p,h in PINS.items():
  if sha(p)!=h:raise RuntimeError(f"510 source changed: {p}")
if __name__=="__main__":main()
