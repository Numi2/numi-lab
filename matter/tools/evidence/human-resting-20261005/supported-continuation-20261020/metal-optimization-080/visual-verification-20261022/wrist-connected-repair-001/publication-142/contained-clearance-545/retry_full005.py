from pathlib import Path
import importlib.util
W=Path(__file__).parent;sp=importlib.util.spec_from_file_location('continuation545',W/'continue545.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
O=W/'full-point-005';args=m.read(O/'fit.launch.json')['argv'][2:]
for flag,value in [('--output',str(O/'retry-fit')),('--area-scp-iterations','4'),('--trust-mm','12')]:args[args.index(flag)+1]=value
m.run(args,O/'retry-fit',True)
