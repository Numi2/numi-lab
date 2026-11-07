#!/usr/bin/env python3
"""Compare retained fixed-area native traces with independent CPU FEM RK4."""
import json, pathlib, sys
p=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'docs/evidence/contact-area-reference-20261107')
read=lambda name:json.loads((p/name).read_text())
ref=[read('fp64-fem-h'+name+'us.json') for name in ('2','1','0.5')]
native=read('temporal-refinement.json')['runs']
def delta(c,f):
    force=max(abs(a-b) for a,b in zip(c['force_bins_N'],f['force_bins_N']))/max(map(abs,f['force_bins_N']))
    impulse=abs(c['impulse_Ns']-f['impulse_Ns'])/abs(f['impulse_Ns'])
    return {'binned_force_relative_change':force,'integrated_impulse_relative_change':impulse}
for x in ref:
    assert x['nodes']==27 and x['tetrahedra']==48
    assert abs(x['mass_kg']-.0088)<1e-15 and abs(x['area_m2']-.0004)<1e-15
    x['global_momentum_error_Ns']=abs(x['final_momentum_Ns']-(x['impulse_Ns']-x['mass_kg']*9.81*x['duration']))
    assert x['global_momentum_error_Ns']<1e-12
pairs=[dict(coarse_h=c['h'],fine_h=f['h'],**delta(c,f)) for c,f in zip(ref,ref[1:])]
results=[]
for row in native:
    x={'force_bins_N':row['force_bins_N'],'impulse_Ns':row['total_impulse_Ns']}
    results.append(dict(h=row['h'],**delta(x,ref[-1])))
valid=pairs[-1]['binned_force_relative_change']<=.0025 and pairs[-1]['integrated_impulse_relative_change']<=.0025
print(json.dumps({'schema':'numi.contact.fp64-fem-comparison.v1',
 'reference_integration':'independent CPU FP64 explicit RK4, fixed mesh and regularized law',
 'reference_refinement':pairs,'reference_fixed_mesh_temporal_gate':valid,
 'native_vs_fine_reference':results,
 'reference_momentum_error_Ns':[x['global_momentum_error_Ns'] for x in ref],
 'limits':['FEM semidiscrete model parity is based on independently coded fixture parameters, not bitwise matching compiled GPU constants',
           'temporal convergence is only established for the 2-division mesh and chosen regularized law',
           'mesh convergence, hard-contact limit and physical material validation remain open',
           'the native-vs-reference differences do not identify numerical damping as the only cause']},indent=2))
raise SystemExit(0 if valid else 1)
