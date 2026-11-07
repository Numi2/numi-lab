#!/usr/bin/env python3
"""Analyze fixed-law surface traces and same-mesh quadrature comparison."""
import json,pathlib,sys
e=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'docs/evidence/contact-surface-diagnosis-20261107')
old=pathlib.Path('docs/evidence/contact-mesh-reference-20261107')
area=pathlib.Path('docs/evidence/contact-area-reference-20261107')
def load(n,h,mode):return json.loads((e/f'n{n}-h{h}-{mode}.json').read_text())
def diff(a,b):
 return {'peak_normalized_binned_force':max(abs(x-y) for x,y in zip(a['force_bins_N'],b['force_bins_N']))/max(b['force_bins_N']),
         'relative_impulse':abs(a['impulse_Ns']-b['impulse_Ns'])/abs(b['impulse_Ns'])}
rows=[]
for n in (2,4,8,16):
 x=load(n,'5e-07','nodal')
 y=json.loads(((area/'fp64-fem-h0.5us.json') if n==2 else (old/f'n{n}-h5e-07.json')).read_text())
 assert x['force_bins_N']==y['force_bins_N'] and x['impulse_Ns']==y['impulse_Ns']
 assert abs(x['mass_kg']-.0088)<1e-14 and abs(x['area_m2']-.0004)<1e-14
 peak=max(x['spatial_trace'],key=lambda q:q[1]);rms=max(x['spatial_trace'],key=lambda q:q[4])
 v=max(x['spatial_trace'],key=lambda q:q[8])
 rows.append({'n':n,'previous_outputs_exact':True,'episodes_s':x['contact_episodes_s'],
  'peak_sample_time_s':peak[0],'peak_sample_force_N':peak[1],
  'corner_force_at_peak_N':peak[2],'edge_force_at_peak_N':peak[3],
  'max_displacement_rms_m':rms[4],'time_of_max_rms_s':rms[0],
  'cosine_modes_at_max_rms_m':rms[5:8],'max_surface_velocity_rms_mps':v[8]})
a,b=load(8,'1e-06','gaussian'),load(8,'5e-07','gaussian')
n=load(8,'5e-07','nodal');temporal=diff(a,b);quadrature=diff(n,b)
additional=None
if (e/'n16-h5e-07-gaussian.json').exists() and (e/'n16-h1e-06-gaussian.json').exists():
 a16=load(16,'1e-06','gaussian');b16=load(16,'5e-07','gaussian');n16=load(16,'5e-07','nodal')
 t16=diff(a16,b16);q16=diff(n16,b16)
 assert t16['peak_normalized_binned_force']<=.0025 and t16['relative_impulse']<=.0025
 assert b16['min_gap_m']>1e-7
 assert abs(b16['final_momentum_Ns']-(b16['impulse_Ns']-b16['mass_kg']*9.81*b16['duration']))<1e-12
 additional={'gaussian_temporal':t16,'same_mesh_nodal_vs_gaussian':q16,
             'gaussian_mesh_n8_to_n16':diff(b,b16),
             'gaussian_minimum_gap_m':b16['min_gap_m'],'gaussian_contact_episodes_s':b16['contact_episodes_s']}
assert temporal['peak_normalized_binned_force']<=.0025 and temporal['relative_impulse']<=.0025
assert b['min_gap_m']>1e-7
assert abs(b['final_momentum_Ns']-(b['impulse_Ns']-b['mass_kg']*9.81*b['duration']))<1e-12
print(json.dumps({'schema':'numi.contact.surface-diagnosis.v1','nodal_meshes':rows,
 'gaussian_temporal':temporal,'same_mesh_nodal_vs_gaussian':quadrature,
 'n16_extension':additional,
 'gaussian_minimum_gap_m':b['min_gap_m'],'gaussian_contact_episodes_s':b['contact_episodes_s'],
 'limits':['RMS and cosine projections are surface diagnostics, not a complete FEM eigenmode decomposition',
 'corner and edge nodal support areas shrink with mesh refinement',
 'both quadratures discretize the same regularized pressure energy but neither is a converged continuum reference',
 '0.1ms point samples do not locate force peaks between samples']},indent=2))
