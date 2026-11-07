#!/usr/bin/env python3
"""Separate temporal checks from fixed-law fixed-duration mesh comparisons."""
import json,pathlib,sys
root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'docs/evidence/contact-mesh-reference-20261107')
previous=pathlib.Path('docs/evidence/contact-area-reference-20261107')
def load(n,h):
 p=previous/('fp64-fem-h'+('1' if h==1e-6 else '0.5')+'us.json') if n==2 else root/('n'+str(n)+'-h'+str(h)+'.json')
 x=json.loads(p.read_text());assert x['mesh_divisions']==n and x['h']==h
 assert abs(x['area_m2']-.0004)<1e-14 and abs(x['mass_kg']-.0088)<1e-14
 x['momentum_error_Ns']=abs(x['final_momentum_Ns']-(x['impulse_Ns']-x['mass_kg']*9.81*x['duration']))
 return x
def diff(a,b):
 return dict(binned_force_relative_change=max(abs(x-y) for x,y in zip(a['force_bins_N'],b['force_bins_N']))/max(b['force_bins_N']),
  integrated_impulse_relative_change=abs(a['impulse_Ns']-b['impulse_Ns'])/abs(b['impulse_Ns']))
temporal=[];fine={}
meshes=(2,4,8,16) if (root/'n16-h5e-07.json').exists() else (2,4,8)
for n in meshes:
 a,b=load(n,1e-6),load(n,5e-7);fine[n]=b
 d=diff(a,b)
 temporal.append(dict(mesh=n,**d,temporal_gate=d['binned_force_relative_change']<=.0025 and d['integrated_impulse_relative_change']<=.0025,
  clearance_gate=b['min_gap_m']>1e-7,momentum_gate=b['momentum_error_Ns']<1e-12,
  min_gap_m=b['min_gap_m'],contact_episodes_s=b['contact_episodes_s'],impulse_Ns=b['impulse_Ns']))
spatial=[]
for n,m in zip(meshes,meshes[1:]):
 d=diff(fine[n],fine[m])
 spatial.append(dict(coarse_mesh=n,fine_mesh=m,**d,spatial_gate=d['binned_force_relative_change']<=.0025 and d['integrated_impulse_relative_change']<=.0025))
print(json.dumps({'schema':'numi.contact.fp64-fem-mesh.v1','temporal':temporal,'spatial':spatial,
 'temporal_gates_all':all(x['temporal_gate'] and x['clearance_gate'] and x['momentum_gate'] for x in temporal),
 'spatial_gate':all(x['spatial_gate'] for x in spatial),
 'limits':['same finite activation width and fixed area energy density; this is not a hard-contact limit',
 'each spatial refinement changes both FEM approximation and surface quadrature',
 'not a physical material validation or native Metal performance measurement']},indent=2))
