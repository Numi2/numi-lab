#!/usr/bin/env python3
"""Run the predeclared, descending step ladder, retaining each attempt."""
import json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parents[2]
p=root/'docs/evidence/midpoint-spatial-cost-dynamic-20261108'
b=pathlib.Path('/Users/numi/numi-matter-midpoint-reference-build-20261108')
meshes=[int(x) for x in sys.argv[1:]] or [2,4,8]
ladder=[('125','0.000125'),('0625','0.0000625'),('03125','0.00003125'),('015625','0.000015625'),('0078125','0.0000078125'),('00390625','0.00000390625')]
for n in meshes:
 for arm in ['ipc_area_midpoint','ipc_area']:
  selected=None
  for tag,h in ladder:
   name=f'native-n{n}-{arm}-h{tag}'
   path=p/(name+'.json');log=p/(name+'.log');analysis=p/(name+'-analysis.json')
   if path.exists() or log.exists() or analysis.exists():
    raise RuntimeError('attempt exists; inspect before recovery: '+name)
   cmd=[str(b/'metalrobo_plane_contact_probe'),str(b/'NumiMatter.metallib'),h,str(round(.01/float(h))),str(n),arm,'0.0002','physical','contact','reference']
   start=time.monotonic()
   with path.open('w') as out,log.open('w') as err:
    proc=subprocess.run(cmd,stdout=out,stderr=err)
   try:
    check=subprocess.run(['python3',str(root/'tests/contact/native_trace_analysis.py'),str(path),str(p/f'rk4-n{n}-h025.json')],capture_output=True,text=True)
    analysis.write_text(check.stdout)
    if check.stderr:(p/(name+'-analysis.log')).write_text(check.stderr)
    passed=proc.returncode==0 and check.returncode==0
   except Exception as exc:
    (p/(name+'-analysis.log')).write_text(repr(exc));passed=False
   print(name,'probe',proc.returncode,'pass',passed,'elapsed_s',round(time.monotonic()-start,1),flush=True)
   if passed:selected=h;break
  print('selected',n,arm,selected,flush=True)
