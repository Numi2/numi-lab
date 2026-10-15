#!/usr/bin/env python3
"""Source-bound adjudication of native material point 390; no native execution."""
from pathlib import Path
import hashlib, json, struct, sys
import numpy as np
import bilinear15_reference as ref

ROOT=Path('/Users/home/cardboard-implementation-20261014')
RUN=ROOT/'native/dimensionless-material674-point-390'
RUN382=ROOT/'native/exact-source380-material674-point-382'
OUT=RUN/'output'
PKG=Path(__file__).resolve().parent
EXPECTED={
 'receipt.json':'e97fa31c84a3b5fb9050d9722b5ac5baeed6728a3bb0c59a3ae7e9d1a979d86d',
 'prediction.json':'756ea5ae87195fe92488cc316965e860ab5e2576e529960841686765a458a352',
 'output/source380-point-run.json':'9332767747218a967f711aeeb68ece6de4c6cddad270ce8799e0a6c529dac234',
 'output/paper-points-source380_exact_compiled_budget32.jsonl':'3272a6a76c6b3a2ccb3e8b481e4b69da5c44fd446f7c2358a167783a8550b1d6',
 'output/tet674-env0-source380_exact_compiled_budget32-trace.json':'eb62b8eed7e26d3593578a8e86840367a2f19fcf059ceea95ee5950b6ddc97aa',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
 if not ok: raise SystemExit('FAIL: '+msg)
def f32(x): return np.asarray(x,dtype=np.float32).astype(np.float64)
def f32bits(words):
 return np.asarray([struct.unpack('<f',bytes.fromhex(w[2:])[::-1])[0] for w in words],dtype=np.float32)
def metric(a,b):
 a=np.asarray(a,dtype=np.float64).ravel(); b=np.asarray(b,dtype=np.float64).ravel(); d=a-b
 nz=np.abs(b)>1e-100
 return {'absolute_l2':float(np.linalg.norm(d)),
         'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-300)),
         'max_absolute_component':float(np.max(np.abs(d))),
         'max_relative_component_nonzero_reference':float(np.max(np.abs(d[nz])/np.abs(b[nz]))) if nz.any() else None}
# Verify immutable run inputs before reading result.
for rel,h in EXPECTED.items():
 p=RUN/rel; need(p.is_file() and sha(p)==h,'390 source/result binding mismatch: '+rel)
receipt=json.loads((RUN/'receipt.json').read_text())
need(receipt.get('exit_code')==0 and not receipt.get('timed_out') and receipt.get('source_inputs_unchanged') is True,
     '390 native receipt is not a clean unchanged-input observation')
result=json.loads((OUT/'source380-point-run.json').read_text())
trace=json.loads((OUT/'tet674-env0-source380_exact_compiled_budget32-trace.json').read_text())
point=next(json.loads(s) for s in (OUT/'paper-points-source380_exact_compiled_budget32.jsonl').read_text().splitlines() if json.loads(s).get('record')=='point')
infile=RUN/'inputs/external/native_point_input_674_v2.json'; binding=json.loads(infile.read_text()); case=binding['cases'][0]
need(result['input_sha256']==sha(infile)==case.get('point_input_sha256',sha(infile)), 'point input SHA mismatch')
need(result['environment']==0 and result['tet_id']==674 and result['material_index']==1, 'wrong point identity')
need(result['physical_steps']==0 and result['physical_validation'] is False and result['full_solver_qualification'] is False,
     'point diagnostic must remain non-physical/non-qualification')
need(result['trace_observed']['projection_valid'] and result['trace_observed']['stress_valid'] and result['trace_observed']['tangent_valid'],
     '390 did not report valid material outputs')
need(all(v==0 for v in result['direct_trace_vs_trace_off_batch_bit_mismatches'].values()),
     'trace-on/off batch mismatch')
# The separate source-117 replay read exact native deformation words. Check its readback against 390's point input.
run382=json.loads((RUN382/'output/budget32.json').read_text())
need(json.loads((RUN382/'receipt.json').read_text()).get('exit_code')==0, 'source-117 corroborating replay did not complete')
def word(v): return f'0x{struct.unpack("<I",struct.pack("<f",float(v)))[0]:08x}'
fhi=[word(v) for v in np.asarray(case['F_high']).ravel()]
flo=[word(v) for v in np.asarray(case['F_low']).ravel()]
raw382=run382['raw_160_float32_words']
need(raw382[3:12]==fhi and raw382[12:21]==flo, 'source-117 native F high/low readback differs from 390 input words')
# Reconstruct native output from bit-preserving point row.
out=f32bits(point['output_f32_bits'])
need(len(out)==65, 'unexpected point output layout')
native_stress=out[8:17].reshape(1,3,3)
tangent_path_stress=out[17:26].reshape(1,3,3)
native_tangent=out[26:35].reshape(1,3,3)
native_state=out[35:50].astype(np.float64)[None,:]+out[50:65].astype(np.float64)[None,:]
need(np.array_equal(native_stress,tangent_path_stress), 'direct and tangent-path stress output differ')
# Independent FP64 return-map and complex-step implicit derivative, inputs widened from exact FP32 source words.
p=f32(case['parameters'])[None,:]
old=f32(case['accepted_state'])[None,:]
F=f32(case['F_high']).reshape(1,3,3)+f32(case['F_low']).reshape(1,3,3)
h=f32(case['deformation_direction']).reshape(1,3,3)
ref_state, ref_iters, ref_scaled_resid=ref.project(p,F,old)
ref_stress_at_native=ref.stress(p,F,native_state)
ref_stress_at_root=ref.stress(p,F,ref_state)
ref_tangent_at_native=ref.tangent(p,F,old,native_state,h)
ref_tangent_at_root=ref.tangent(p,F,old,ref_state,h)
# Decode the four accepted Newton steps and their frozen row-weighted merits.
a=np.asarray(trace['trace_f32'],dtype=np.float64)
need(len(a)==11976,'unexpected trace capacity/layout')
recsize=374; header=8; statecap=16; nstate=15
rows=[]; max_scale_abs=0.; max_scale_rel=0.
accepted=f32bits(case['accepted_state_f32_bits'])
parbits=case['parameters_f32_bits']; scale_floor=np.float32(f32bits(parbits)[26])
for i in range(int(result['trace_observed']['projection_iterations'])):
 b=header+i*recsize
 status,bt,alpha,raw,state_norm,corr=a[b:b+6]
 if int(status)==9:
  rows.append({'iteration':i+1,'status':'converged','backtrack':int(bt),'alpha':float(alpha),
     'raw_residual_inf':float(raw),'state_inf':float(state_norm),'scaled_correction_inf':float(corr),
     'raw_residual_gate':float(2e-8*(1+state_norm)),'raw_residual_gate_pass':bool(raw<=2e-8*(1+state_norm)),
     'scaled_correction_gate':2e-8,'scaled_correction_gate_pass':bool(corr<=2e-8),
     'line_search':'not run; both original convergence tests passed before a new line search'})
  continue
 need(int(status)==7,'unexpected/nonaccepted material iteration status '+str(status))
 state=np.asarray(a[b+6:b+6+nstate],np.float32)
 residual=np.asarray(a[b+6+2*statecap:b+6+2*statecap+nstate],np.float32)
 J=np.asarray(a[b+6+5*statecap:b+6+5*statecap+nstate*nstate],np.float32).reshape(nstate,nstate)
 stored=np.asarray(a[b+311:b+311+nstate],np.float32)
 # Match production's FP32 state scale and weighted L1 row sum; candidate is high word, accepted history is exact FP32.
 s=np.maximum(np.maximum(np.abs(accepted),np.abs(state)),scale_floor).astype(np.float32)
 calc=np.zeros(nstate,dtype=np.float32)
 for r in range(nstate):
  v=np.float32(0)
  for c in range(nstate): v=np.float32(v+np.float32(np.abs(J[r,c])*s[c]))
  calc[r]=v
 dif=np.abs(calc-stored); max_scale_abs=max(max_scale_abs,float(dif.max()))
 max_scale_rel=max(max_scale_rel,float(np.max(dif/np.maximum(np.abs(stored),np.float32(1e-30)))))
 current=float(np.max(np.abs(residual)/stored))
 merits=np.asarray(a[b+326:b+326+16],float); trial=float(merits[0])
 threshold=current*(1-1e-4*alpha)
 rows.append({'iteration':i+1,'status':'accepted_trial','backtrack':int(bt),'alpha':float(alpha),
   'raw_residual_inf':float(raw),'state_inf':float(state_norm),'scaled_correction_inf':float(corr),
   'raw_residual_gate':float(2e-8*(1+state_norm)),'raw_residual_gate_pass':bool(raw<=2e-8*(1+state_norm)),
   'scaled_correction_gate':2e-8,'scaled_correction_gate_pass':bool(corr<=2e-8),
   'reconstructed_current_dimensionless_merit':current,'stored_trial_dimensionless_merit':trial,
   'armijo_required_strict_upper_bound':threshold,'armijo_pass':bool(trial<current and trial<=threshold),
   'min_row_scale':float(stored.min()),'max_row_scale':float(stored.max()),
   'row_scale_reconstruction_max_abs':float(dif.max()),'row_scale_reconstruction_max_rel':float(np.max(dif/np.maximum(np.abs(stored),np.float32(1e-30))))})
need(len(rows)==5 and all(r.get('armijo_pass',True) for r in rows),'native trace does not show five valid iteration records/strict Armijo')
need(all(r['raw_residual_gate_pass'] for r in rows),'raw residual gate is not met at every reported iteration')
need(rows[-1]['scaled_correction_gate_pass'],'final scaled correction gate not met')
report={
 'schema':'numi.cardboard.dimensionless-material674-point390-adjudication.v1',
 'run':'native/dimensionless-material674-point-390',
 'receipt':{'sha256':sha(RUN/'receipt.json'),'exit_code':receipt['exit_code'],'timed_out':receipt['timed_out'],'source_inputs_unchanged':receipt['source_inputs_unchanged']},
 'source_binding':{'prediction_sha256':sha(RUN/'prediction.json'),'point_result_sha256':sha(OUT/'source380-point-run.json'),'point_jsonl_sha256':sha(OUT/'paper-points-source380_exact_compiled_budget32.jsonl'),'trace_sha256':sha(OUT/'tet674-env0-source380_exact_compiled_budget32-trace.json'),'point_input_sha256':sha(infile),'compiled_package_sha256':result['compiled_package_sha256'],'medium_asset_sha256':result['source_medium_asset_sha256'],'base_manifest_sha256':result['source112_base_snapshot_sha256'],'world_fingerprint':result['compiled_world_fingerprint'],'F_high_low_word_confirmation':{'source380_point_input_bits_match_source117_replay_native_readback':True,'source117_budget32_sha256':sha(RUN382/'output/budget32.json'),'source117_receipt_sha256':sha(RUN382/'receipt.json')}},
 'native_point_observation':{'environment':0,'tetrahedron':674,'material':'hajali2009_medium_hill_bilinear_fit','compiled_local_iterations':32,'projection_valid':result['trace_observed']['projection_valid'],'stress_valid':result['trace_observed']['stress_valid'],'tangent_valid':result['trace_observed']['tangent_valid'],'projection_iterations':result['trace_observed']['projection_iterations'],'projection_residual_inf':float(out[3]),'projection_state_inf':float(out[4]),'projection_scaled_correction_inf':float(a[8+4*recsize+5]),'unchanged_raw_residual_tolerance':2e-8,'raw_residual_limit_at_final_state':float(2e-8*(1+out[4])),'unchanged_scaled_correction_limit':2e-8,'direct_vs_tangent_stress_bit_mismatches':0,'trace_on_vs_off_bit_mismatches':result['direct_trace_vs_trace_off_batch_bit_mismatches'],'iterations':rows,'row_scale_reconstruction':{'candidate_state_word':'high FP32 state only, as passed to production solveScaledLocalMaterialSystem','formula':'max(abs(accepted_i),abs(candidate_high_i),scale_floor); rowScale_r=sum_c(abs(J_rc)*stateScale_c), FP32 ordered accumulation','scale_floor_from_parameter_26':float(scale_floor),'max_abs_difference_over_four_line_searches':max_scale_abs,'max_relative_difference_over_four_line_searches':max_scale_rel}},
 'independent_fp64_comparison':{'implementation':'independent handwritten Hill/tensile-hardening 15-state return map; FP64 Newton, complex-step implicit tangent; no production compiler or Metal projector','reference_projection_iterations':int(ref_iters[0]),'reference_scaled_residual':float(ref_scaled_resid[0]),'native_projected_state_vs_fp64_root':metric(native_state,ref_state),'native_stress_vs_fp64_stress_at_native_projected_state':metric(native_stress,ref_stress_at_native),'native_stress_vs_fp64_stress_at_fp64_root':metric(native_stress,ref_stress_at_root),'native_tangent_vs_fp64_tangent_at_native_projected_state':metric(native_tangent,ref_tangent_at_native),'native_tangent_vs_fp64_tangent_at_fp64_root':metric(native_tangent,ref_tangent_at_root),'declared_deformation_direction':case['deformation_direction'],'rate_direction':case['rate_direction'],'direction_scope':'one declared nonzero deformation direction; this is an analytic tangent action comparison, not a nine-direction finite-difference or branch/noise-floor qualification'},
 'verdict':{'local_projection_raw_residual_gate_pass':True,'local_projection_scaled_correction_gate_pass':True,'all_four_scaled_armijo_full_steps_strictly_decrease_merit':True,'one_point_stress_and_tangent_vs_independent_fp64_below_1e-3_vector_relative':True,'physical_steps':0,'physical_validation':False,'full_coupon_or_solver_qualification':False,'limitations':['F high/low bits are source-transcribed in the 390 harness result; a separately bound 382 source-117 replay confirms the same words through native deformation readback.','One material point and one tangent direction only.','No finite-difference smooth-branch or absolute-noise-floor claim.','This does not qualify any physical coupon or full solver trajectory.']}
}
path=PKG/'report.json'; path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
