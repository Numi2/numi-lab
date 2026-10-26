"""Retain focused local CPU tests with exact source pins for publication."""
from pathlib import Path
import importlib.util,sys,json
P=Path(__file__).resolve().parent;N=P.parent
sp=importlib.util.spec_from_file_location('v6teststage',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
files=[P/n for n in ['quality_policy025_variant.py','run_quality_policy025_variant.py','run_chain_and_check_v5.py','test_root_policy_v5.py','test_quality_policy025_variant.py']]
r=m.run_stage([sys.executable,'-m','pytest','-q',P/'test_root_policy_v5.py',P/'test_quality_policy025_variant.py'],P/'tests-publication-001',slot=False,source_paths=(Path(__file__),*files),timeout=60)
print(json.dumps(r))

