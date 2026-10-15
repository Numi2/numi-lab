#!/usr/bin/env python3
"""Reproducible CPU-only geometry hypothesis for captured source103 pair 92/127."""
import hashlib, json, math, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path('/Users/home/cardboard-implementation-20261014')
PKG = ROOT / 'analysis/source103-pair-ccd-diagnostic-003'
OUT = Path(__file__).with_name('report.json')
sys.path.insert(0, str(PKG))
from prepare_pair_inputs import closest_case

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bits(x): return [f'{int(v):08x}' for v in np.asarray(x,dtype='<f4').view('<u4').reshape(-1)]
def norm(v): return math.sqrt(math.fsum(float(x)*float(x) for x in v))
def f32(x): return float(np.float32(x))
def cpu_trace(case, maxiter=64):
    start=np.asarray(case['points_start_m'],dtype=np.float64).reshape(4,3)
    finish=np.asarray(case['points_finish_m'],dtype=np.float64).reshape(4,3)
    thickness=float(np.float32(case['thickness_m']))
    eps=float(np.float32(1.1920928955078125e-7))
    scale=max(float(np.max(np.abs(start))),float(np.max(np.abs(finish))))
    # Mirror the source expression in FP32 for the effective threshold; CPU geometry remains FP64.
    pad=float(np.float32(np.float32(8.0)*np.float32(eps)*np.float32(max(thickness,scale,1.0e-12))))
    effective=float(np.float32(np.float32(thickness)+np.float32(pad)))
    displacements=finish-start
    # Prepared order is left-edge endpoint 0/1 followed by right-edge endpoint 0/1.
    speed=max(norm(displacements[0]),norm(displacements[1])) + max(norm(displacements[2]),norm(displacements[3]))
    relative=max(norm(displacements[i]-displacements[j]) for i in (0,1) for j in (2,3))
    def dist(t):
        x=start+t*(finish-start)
        return closest_case(1,x)[0]
    rows=[]; t=0.0; hit=None
    for i in range(maxiter):
        d=dist(t); rows.append({'iteration':i,'time':t,'distance_m':d,'gap_to_effective_m':d-effective})
        if d <= effective:
            hit=i; break
        step=max((d-effective)/speed,1.0e-6)
        t+=step
        if t > 1.0 + 8.0*eps: break
        t=min(t,1.0)
    def threshold_crossing(level):
        if dist(0.0) <= level: return 0.0
        if dist(1.0) > level: return None
        lo,hi=0.0,1.0
        for _ in range(80):
            mid=0.5*(lo+hi)
            if dist(mid) <= level: hi=mid
            else: lo=mid
        return hi
    return {'nominal_thickness_m':thickness,'roundoff_pad_f32_m':pad,
            'effective_thickness_f32_m':effective,'effective_thickness_f32_bits':bits([effective]),
            'production_source_bound_promoted_estimate_m':speed,
            'relative_geometry_lipschitz_bound_fp64_m':relative,
            'absolute_over_relative_ratio':speed/relative if relative else None,
            'sampled_start_distance_m':dist(0.0),'sampled_finish_distance_m':dist(1.0),
            'sampled_min_m':min(dist(i/4096) for i in range(4097)),
            'first_time_within_nominal_thickness_fp64':threshold_crossing(thickness),
            'first_time_within_effective_thickness_fp64':threshold_crossing(effective),
            'ca_trace_fp64_geometry_using_source_bound':rows,
            'first_hit_iteration_in_extended_cpu_trace':hit,
            'native_ccd_result':'NOT_OBSERVED: package 003 was compile/CPU verified only; previous native 364 failed before dispatch',
            'interpretation':'Independent FP64 closest-segment geometry over exact promoted captured FP32 endpoint words. This is not a production Metal trace or certificate.'}

manifest=json.loads((PKG/'prepared/prepared.json').read_text())
selected={}
for capture in ('center359','trial358'):
    case=next(c for c in manifest['cases'] if c['name']==f'{capture}_ee_edge0_edge0')
    selected[capture]={'case_name':case['name'],'captured_endpoint_float32_bits':case['points_start_f32_bits']+case['points_finish_f32_bits'],
        'pair':case['primitive_pair'],'candidate_identity':case['candidate_identity'],
        'dt_seconds_f32_bits':case['dt_f32_bits'],'thickness_f32_bits':case['record_f32_bits'][1],
        'primitive_nodes':case['primitive_nodes'],'geometry':cpu_trace(case)}
source=ROOT/'snapshots/source-whole-mortar-proof-103/matter/src/metal/contact.metalinc'
common=ROOT/'snapshots/source-whole-mortar-proof-103/matter/src/metal/common.metalinc'
cap=ROOT/'native/first-fatal-private-trial-358/private-trial.coupled.json'
capbin=ROOT/'native/first-fatal-private-trial-358/private-trial.coupled.bin'
report={
 'schema':'numi.cardboard.source103_pair_geometry_hypothesis.v1',
 'verdict':'CPU_GEOMETRY_SUGGESTS_NEAR_THICKNESS_CROSSING; NATIVE_SUBCALL_UNOBSERVED',
 'scope':'Read-only CPU hypothesis for exact captured EE edge0/edge0 endpoints. No source or transaction changes.',
 'bindings':{
  'source_snapshot':str(ROOT/'snapshots/source-whole-mortar-proof-103'),
  'contact_metalinc_sha256':sha(source),'common_metalinc_sha256':sha(common),
  'prepared_manifest_sha256':sha(PKG/'prepared/prepared.json'),'prepared_cases_bin_sha256':sha(PKG/'prepared/cases.bin'),
  'native_trial358_manifest_sha256':sha(cap),'native_trial358_payload_sha256':sha(capbin),
  'trial358_build_source_fingerprint':manifest['capture_inputs'][1]['physics_fingerprint'],
  'trial358_device_program_fingerprint':manifest['capture_inputs'][1]['device_program_fingerprint']},
 'selected_identity':{'pair':[92,127],'candidate_slot':283,'candidate_identity':[92,127,283,1],
  'root':1201,'newton_iteration':7,'private_trial':23,'kind':'edge_edge','feature_indices':[0,0]},
 'cases':selected,
 'source_algorithm':{
  'edge_edge_iterations':17,'speed_bound':'max(|dp0|,|dp1|)+max(|dq0|,|dq1|)',
  'advance':'max((distance-effectiveThickness)/speedBound,1e-6)',
  'effective_thickness':'thickness + 8*float_epsilon*max(thickness,coordinateScale,1e-12)',
  'source_lines':{'common.metalinc':'79-84','contact.metalinc':'3331-3428'},
  'safety_note':'Any replacement bound needs an outward FP32 evaluation-error argument; the relative-motion bound here is exact-real geometry only.'},
 'narrow_followup_proposal':{
  'first_step':'Run package 003 native direct-routine replay; compare 30 direct production outputs with trace clone and retain exact per-call certification/iterations.',
  'if_EE_0_0_exhausts':'Evaluate a bounded conservative-advance continuation for exhausted calls only, carrying the same time and using the exact same speed bound, effective thickness, feature distance and certified miss tests. Continue to fail closed if the tail cannot certify. This changes no threshold and cannot turn a true hit into a miss; it may classify the near-thickness crossing as a hit rather than status-6 uncertainty.',
  'guard':'Keep the current production path unchanged until the native trace demonstrates that this exact subcall exhausts. Do not substitute the smaller Galilean-relative bound without a separate outward-rounding proof.',
  'physical_interpretation':'A certified hit still means the candidate encounters the configured contact band and may be rejected; it is not evidence that the action is collision-free or that a solver defect is fixed.'},
 'limitations':['CPU reference is double precision over promoted exact float32 endpoints, but closest-point arithmetic and the CCD walk are independently implemented.','No successful native package-003 dispatch is included.','The calculation does not identify whether pair 92/127 is an intended interface or an unintended geometric contact.','No collision threshold, iteration limit, source code, or physical state was changed.']}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('report',OUT)
print('sha256',sha(OUT))
for tag,row in selected.items():
 g=row['geometry']; print(tag,'start_um',g['sampled_start_distance_m']*1e6,'finish_um',g['sampled_finish_distance_m']*1e6,'effective_um',g['effective_thickness_f32_m']*1e6,'hit_iter',g['first_hit_iteration_in_extended_cpu_trace'],'source_iter',g['edge_edge_iterations'] if 'edge_edge_iterations' in g else 17,'speedratio',g['absolute_over_relative_ratio'])
