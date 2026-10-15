#!/usr/bin/env python3
"""Check FP64 root against production dynamic correction metric for point 390."""
from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
import bilinear15_reference as ref
ROOT=Path('/Users/home/cardboard-implementation-20261014')
RUN=ROOT/'native/dimensionless-material674-point-390'; OUT=RUN/'output'
REF382=ROOT/'native/exact-source380-material674-point-382'
PKG=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x):return np.asarray(x,dtype=np.float32).astype(np.float64)
def words(ws):return np.asarray([struct.unpack('<f',bytes.fromhex(w[2:])[::-1])[0] for w in ws],dtype=np.float64)
def met(a,b):
 d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64);b=np.asarray(b,dtype=np.float64);nz=np.abs(b)>1e-100
 return {'abs_l2':float(np.linalg.norm(d)),'rel_l2':float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-300)),'max_abs':float(np.max(np.abs(d))),'max_component_rel_nonzero':float(np.max(np.abs(d[nz])/np.abs(b[nz]))) if nz.any() else None}
def gate(z,p,old):
 r=ref.residual(p,F,old,z);J=ref.jacobian(p,F,old,z)
 state_scale=np.maximum(np.maximum(np.abs(old),np.abs(z)),floor)
 row_scale=np.sum(np.abs(J)*state_scale[None,:],axis=1)
 delta=np.linalg.solve(J,-r)
 return {'raw_residual_inf':float(np.max(np.abs(r))),'raw_gate_limit':float(2e-8*(1+np.max(np.abs(z)))),'raw_gate_pass':bool(np.max(np.abs(r))<=2e-8*(1+np.max(np.abs(z)))),'dynamic_scaled_residual_inf':float(np.max(np.abs(r)/row_scale)),'dynamic_scaled_correction_inf':float(np.max(np.abs(delta)/state_scale)),'dynamic_scaled_correction_limit':2e-8,'dynamic_scaled_correction_pass':bool(np.max(np.abs(delta)/state_scale)<=2e-8),'correction_vector':delta.tolist(),'row_scale_min':float(row_scale.min()),'row_scale_max':float(row_scale.max())}

i=json.load(open(RUN/'inputs/external/native_point_input_674_v2.json'))
c=i['cases'][0]
t=json.load(open(OUT/'tet674-env0-source380_exact_compiled_budget32-trace.json'))
pt=next(json.loads(s) for s in open(OUT/'paper-points-source380_exact_compiled_budget32.jsonl') if json.loads(s).get('record')=='point')
params=f32(c['parameters']);old=f32(c['accepted_state']);F=f32(c['F_high']).reshape(3,3)+f32(c['F_low']).reshape(3,3);h=f32(c['deformation_direction']).reshape(3,3);floor=float(np.float32(params[26]))
out=words(pt['output_f32_bits']);z_native=out[35:50]+out[50:65];native_stress=out[8:17].reshape(3,3);native_tangent=out[26:35].reshape(3,3)
z_res,iterations,legacy_norm=ref.project(params[None,:],F[None,:,:],old[None,:]);z0=z_res[0]
g0=gate(z0,params,old);z1=z0+np.asarray(g0.pop('correction_vector'));g1=gate(z1,params,old)
gnative=gate(z_native,params,old)
s_ref=ref.stress(params,F,z1);t_ref=ref.tangent(params,F,old,z1,h)
report={
 'schema':'numi.cardboard.point390.fp64-dynamic-root-addendum.v1',
 'bindings':{'point_input_sha256':sha(RUN/'inputs/external/native_point_input_674_v2.json'),'native_point_run_sha256':sha(OUT/'source380-point-run.json'),'native_trace_sha256':sha(OUT/'tet674-env0-source380_exact_compiled_budget32-trace.json'),'reference_implementation_sha256':sha(PKG/'bilinear15_reference.py'),'point':'source380 env0/tet674 medium, source-117 exact F high/low corroboration in separate run382'},
 'method':'The existing independent FP64 project() stops on its own 2e-10 weighted residual criterion. To compare with production, recompute the production dynamic state scale max(abs(accepted),abs(candidate),parameter[26]), current complex-step J row scales, and next Newton correction. Take one extra FP64 Newton update when this stricter correction check fails; do not alter production tolerances or equations.',
 'initial_reference_return':{'legacy_scaled_residual':float(legacy_norm[0]),'reference_iterations':int(iterations[0]),'production_dynamic_metrics':g0},
 'one_extra_reference_newton_update':{'production_dynamic_metrics':g1,'both_original_gates_pass':bool(g1['raw_gate_pass'] and g1['dynamic_scaled_correction_pass'])},
 'native_candidate_promoted_fp64':{'production_dynamic_metrics':gnative,'both_original_gates_pass':bool(gnative['raw_gate_pass'] and gnative['dynamic_scaled_correction_pass'])},
 'native_point_vs_refined_root':{'state':met(z_native,z1),'stress':met(native_stress,s_ref),'tangent_one_declared_direction':met(native_tangent,t_ref)},
 'refined_reference_vectors':{'parameters_from_exact_fp32_input':params.tolist(),'accepted_state_from_exact_fp32_input':old.tolist(),'F_high':f32(c['F_high']).reshape(3,3).tolist(),'F_low':f32(c['F_low']).reshape(3,3).tolist(),'deformation_direction':h.tolist(),'projected_state':z1.tolist(),'stress_Pa':np.asarray(s_ref).tolist(),'tangent_Pa_per_s':np.asarray(t_ref).tolist(),'temperature_K':float(c['temperature_K']),'state_scale_floor':floor},
 'limits':['This is one local material point and one declared deformation direction.','It is analytic implicit tangent agreement, not nine-direction finite-difference/noise-floor qualification.','No physical step, coupon, material calibration, or solver qualification.']}
(PKG/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(PKG/'fp64-reference-vectors-674.json').write_text(json.dumps({'schema':'numi.cardboard.independent-fp64-material-point-vectors.v1','point':report['refined_reference_vectors'],'source_input_sha256':report['bindings']['point_input_sha256'],'native_material_point_run_sha256':report['bindings']['native_point_run_sha256'],'reference_implementation_sha256':report['bindings']['reference_implementation_sha256'],'method':'Handwritten independent 15-state Hill/tensile-hardening residual; FP64 Newton; complex-step implicit Jacobian and tangent; one extra Newton correction to satisfy production dynamic scaled-correction gate.'},indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
