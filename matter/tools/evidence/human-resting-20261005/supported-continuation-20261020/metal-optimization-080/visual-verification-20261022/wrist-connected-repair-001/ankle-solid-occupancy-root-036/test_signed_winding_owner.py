#!/usr/bin/env python3
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human import common_atlas_skin_clearance as cc,cardiac_cavity_intersections as ci
v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],dtype=np.int64)
records=cc._exact_surface_records(v,f);prepared=ci.prepare_signed_winding(records)
cases={"inside":np.array([.125,.125,.125],np.float32),"outside":np.array([2,2,2],np.float32),"boundary":np.array([0,.125,.125],np.float32)}
out={name:ci.signed_winding_number(cc.float32_point_lattice_key(point),prepared) for name,point in cases.items()}
assert out['inside']['status']=='resolved' and abs(out['inside']['winding_number'])==1
assert out['outside']['status']=='resolved' and out['outside']['winding_number']==0
assert out['boundary']['status']=='boundary'
try:ci.prepare_signed_winding(records[:-1])
except Exception as e:out['open_shell_rejected']=str(e)
else:raise AssertionError('open shell admitted')
p=Path(__file__).with_name('predicate-sanity.json');assert not p.exists();p.write_text(json.dumps({'passed':True,'results':out},indent=2)+'\n');print(json.dumps({'passed':True,'results':out}))
