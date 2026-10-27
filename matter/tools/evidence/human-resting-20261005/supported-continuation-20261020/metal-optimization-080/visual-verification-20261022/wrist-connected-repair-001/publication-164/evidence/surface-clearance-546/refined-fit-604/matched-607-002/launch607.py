import importlib.util,json,shutil
from pathlib import Path
n=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546');d=n/'refined-fit-604/matched-607-002'
p=n/'continue_surface_round546.py';s=importlib.util.spec_from_file_location('stages607',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
argv=json.loads((d/'fit-argv.json').read_text());pins=tuple(Path(x) for x in json.loads((d/'launch-source-files607.json').read_text()))
print('free_before',shutil.disk_usage(d).free,flush=True)
print(m.run_stage(argv,d/'fit-stage',slot=True,source_paths=pins,timeout=1800),flush=True)
print('free_after',shutil.disk_usage(d).free,flush=True)
