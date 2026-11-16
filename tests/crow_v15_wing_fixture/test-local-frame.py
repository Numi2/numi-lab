#!/usr/bin/env python3
"""Check free local-frame invariance and unsupported attachment rollback."""
import json,math,subprocess,sys
from pathlib import Path
if len(sys.argv)!=2: raise SystemExit('usage: test-local-frame.py BUILD_DIR')
b=Path(sys.argv[1]).resolve();p=subprocess.run([str(b/'crow-feather-local-frame'),str(b/'shaders/MetalRobo.metallib')],capture_output=True,text=True)
assert p.returncode==0,(p.returncode,p.stderr)
a=[json.loads(line) for line in p.stdout.splitlines() if line.strip()];d={x['backend']:x for x in a if 'backend' in x}
assert len(d)==6
for mode in ('off','on'):
 assert all(d[f'{mode}_{i}']['success'] for i in range(3))
base=d['on_0'];e0=base['elastic']+base['kinetic'];assert e0>0
for i,offset in ((1,1),(2,-1)):
 x=d[f'on_{i}'];gap=max(math.dist([v[0]-offset,v[1],v[2]],z[:3]) for v,z in zip(x['state'],base['state']));assert gap<1e-7,(i,gap)
 assert abs((x['elastic']+x['kinetic'])/e0-1)<.001
assert max(math.dist(x[:3],y[:3]) for x,y in zip(d['off_0']['state'],base['state']))<1e-7
unsupported=next(x for x in a if x.get('event')=='unsupported_attachment');assert unsupported['status']=='invalid_configuration' and not unsupported['dispatched'] and not unsupported['published'] and unsupported['unchanged']
print('PASS: local-frame free rods are translation invariant; attachments fail closed')
