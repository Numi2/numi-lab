#!/usr/bin/env python3
"""Fail-closed equality/rollback regression; requires compiled native probes."""
import json,math,subprocess,sys
from pathlib import Path
if len(sys.argv)!=2:
    raise SystemExit('usage: test_equality_guard.py BUILD_DIR')
build=Path(sys.argv[1]).resolve(); shader=build/'shaders/MetalRobo.metallib'
def run(exe,*args):
    p=subprocess.run([str(build/exe),str(shader),*args],capture_output=True,text=True)
    rows=[json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    runs=[row for row in rows if row.get('event')=='run']
    if len(runs)!=1: raise AssertionError(f'{exe}: missing run result; exit={p.returncode}: {p.stderr}')
    return p.returncode,runs[0]
for damping in ('0','100'):
    code,r=run('crow-feather-equality-gate',damping)
    assert code!=0 and not r['ok'] and r['status']==11 and r['steps']==0 and r['failed']==1,(damping,code,r)
    assert r['quality_statuses'] and r['quality_statuses'][0]['code']==7
    assert r['quality_statuses'][0]['certificates0'][2]>2e-5
    assert len(r['state'])==9
    for i,row in enumerate(r['state']):
        x=.025*i/8;u=min(i/8,1-i/8);y=.000025*(3*u-4*u*u*u)
        assert math.dist(row[:3],[x,y,0])<1e-8,(damping,i,row)
        assert max(map(abs,row[3:]))<1e-9,(damping,i,row)
code,r=run('crow-feather-predicted-position','1','0')
assert code==0 and r['ok'] and r['status']==0 and r['steps']==1 and r['failed']==0
assert all(abs(row[1])<1e-8 and max(map(abs,row[3:]))<1e-9 for row in r['state'])
print('PASS: invalid equalities rejected and rolled back; stationary control accepted')
