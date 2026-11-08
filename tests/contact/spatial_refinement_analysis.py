#!/usr/bin/env python3
"""Predeclared temporal and spatial force/impulse gates for FP64 traces."""
import json, math, pathlib, sys
p=pathlib.Path(sys.argv[1])
def read(n,dt):
 x=json.loads((p/f'rk4-n{n}-h{dt}.json').read_text())
 assert x['mesh_divisions']==n and x['duration']==.01
 assert len(x['trace'])==5001 and len(x['force_bins_N'])==10
 assert len(x['contact_area_weights_m2'])==(n+1)**3
 assert all(math.isfinite(v) and v>=0 for v in x['contact_area_weights_m2'])
 assert abs(sum(x['contact_area_weights_m2'])-.0004)<1e-14
 assert abs(x['area_m2']-.0004)<1e-14
 return x
def compare(a,b):
 assert len(a['trace'])==len(b['trace'])
 assert all(abs(u[0]-v[0])<1e-12 for u,v in zip(a['trace'],b['trace']))
 peak=b['peak_force_N']; impulse=b['impulse_Ns']
 assert peak>0 and impulse>0
 x={'binned_force':max(abs(u-v) for u,v in zip(a['force_bins_N'],b['force_bins_N']))/max(b['force_bins_N']),
    'total_impulse':abs(a['impulse_Ns']-impulse)/impulse,
    'peak_force':abs(a['peak_force_N']-peak)/peak,
    'cumulative_impulse':max(abs(u[2]-v[2]) for u,v in zip(a['trace'],b['trace']))/impulse}
 x['passes']=x['binned_force']<=.0025 and x['total_impulse']<=.0025 and x['peak_force']<=.01 and x['cumulative_impulse']<=.01
 return x
r={(n,t):read(n,t) for n in (2,4,8) for t in ('050','025')}
temporal={str(n):compare(r[n,'050'],r[n,'025']) for n in (2,4,8)}
spatial={f'{a}_to_{b}':compare(r[a,'025'],r[b,'025']) for a,b in ((2,4),(4,8))}
print(json.dumps({'temporal':temporal,'spatial':spatial,
 'reference_area_m2':{str(n):r[n,'025']['area_m2'] for n in (2,4,8)},
 'peak_force_N':{str(n):r[n,'025']['peak_force_N'] for n in (2,4,8)},
 'impulse_Ns':{str(n):r[n,'025']['impulse_Ns'] for n in (2,4,8)},
 'spatial_qualified':temporal['4']['passes'] and temporal['8']['passes'] and spatial['4_to_8']['passes']},indent=2))
