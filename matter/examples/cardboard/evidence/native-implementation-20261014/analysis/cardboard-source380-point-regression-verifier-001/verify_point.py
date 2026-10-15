#!/usr/bin/env python3
"""Opt-in verifier for the exact source-380 env0/tet674 native point diagnostic."""
from pathlib import Path
import argparse, hashlib, json, math, struct, sys
import numpy as np

EXPECTED_INPUT_SHA='7925a6eba993e6bf8b4f536352027d18c221f95c215fe73d1b20ebaec6aeb797'
class VerificationError(RuntimeError): pass

def need(ok,msg):
 if not ok: raise VerificationError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def f32_words(values):
 return [f'0x{struct.unpack("<I",struct.pack("<f",float(v)))[0]:08x}' for v in values]
def floats_from_bits(bits):
 return np.asarray([struct.unpack('<f',bytes.fromhex(w[2:])[::-1])[0] for w in bits],dtype=np.float64)
def vector_metric(actual,expected):
 a=np.asarray(actual,dtype=np.float64).reshape(-1);e=np.asarray(expected,dtype=np.float64).reshape(-1)
 need(a.shape==e.shape,'reference vector dimension mismatch')
 d=np.abs(a-e); denom=np.linalg.norm(e)
 nz=np.abs(e)>1e-100
 return {'relative_l2':float(np.linalg.norm(d)/max(float(denom),1e-300)),
         'max_absolute_component':float(np.max(d)),
         'max_relative_component_nonzero':float(np.max(d[nz]/np.abs(e[nz]))) if nz.any() else None}
def verify(*,fixture,package,medium_asset,reference_vectors,receipt,result,points,trace,f_readback,f_readback_manifest):
 fixture=Path(fixture); package=Path(package); medium_asset=Path(medium_asset)
 reference_vectors=Path(reference_vectors);receipt=Path(receipt);result=Path(result);points=Path(points);trace=Path(trace);f_readback=Path(f_readback);f_readback_manifest=Path(f_readback_manifest)
 need(sha(fixture)==EXPECTED_INPUT_SHA,'origin point fixture SHA mismatch')
 binding=json.loads(fixture.read_text());need(binding.get('schema')=='numi.cardboard.source380-native-material-point-input.v2','unexpected fixture schema')
 need(binding.get('case_count')==1 and len(binding.get('cases',[]))==1,'expected exactly one origin point')
 c=binding['cases'][0];run=binding['run_binding'];desc=run['material_descriptor_expectation']
 need((c['environment'],c['tet_id'],c['material_index'])==(0,674,1),'origin point identity changed')
 need(sha(package)==run['compiled_package_sha256'],'compiled package bytes do not match fixture binding')
 need(sha(medium_asset)==run['source_medium_asset_sha256'],'medium asset bytes do not match fixture binding')
 rcp=json.loads(receipt.read_text());need(rcp.get('exit_code')==0 and not rcp.get('timed_out') and rcp.get('source_inputs_unchanged') is True,'native run receipt is not clean')
 res=json.loads(result.read_text())
 need(res.get('input_sha256')==EXPECTED_INPUT_SHA,'native result is not bound to the origin point input')
 need(res.get('compiled_package_sha256')==run['compiled_package_sha256'],'native result package hash differs from fixture')
 need(res.get('source_medium_asset_sha256')==run['source_medium_asset_sha256'],'native result medium asset hash differs from fixture')
 need(str(res.get('compiled_world_fingerprint'))==str(run['compiled_world_fingerprint']),'native result world fingerprint differs from fixture')
 need((res.get('environment'),res.get('tet_id'),res.get('material_index'))==(0,674,1),'native result point identity mismatch')
 need(res.get('physical_steps')==0 and res.get('physical_validation') is False and res.get('full_solver_qualification') is False,'point result must remain diagnostic-only')
 need(res.get('compiled_material',{}).get('flags')==desc['flags'] and res['compiled_material'].get('state_count')==desc['state_count'] and res['compiled_material'].get('parameter_count')==desc['parameter_count'] and res['compiled_material'].get('local_newton_iterations')==desc['local_newton_iterations'],'compiled material descriptor mismatch')
 need(sha(package)==res['compiled_package_sha256'] and sha(medium_asset)==res['source_medium_asset_sha256'],'actual source bytes differ from native result binding')
 arts=res.get('source380_artifacts',{});ident=run['source_candidate_identity']
 need(arts.get('candidate_nodes_sha256')==ident['candidate_nodes_sha256'] and arts.get('accepted_history_sha256')==ident['accepted_history_sha256'] and arts.get('mesh_sha256')==ident['mesh_sha256'],'candidate/history/mesh provenance differs from origin fixture')
 need(res.get('raw_input_f32_bits')==c['packed_input_f32_bits'],'native packed input words differ from exact fixture')
 need(res.get('raw_accepted_state_f32_bits')==c['accepted_state_f32_bits'],'native accepted-history words differ from exact fixture')
 need(res.get('trace_observed',{}).get('projection_valid') is True and res.get('trace_observed',{}).get('stress_valid') is True and res.get('trace_observed',{}).get('tangent_valid') is True,'native projection/stress/tangent status is invalid')
 mism=res.get('direct_trace_vs_trace_off_batch_bit_mismatches',{})
 need(set(mism)=={'stress_direct','stress_tangent_path','tangent','projected_state_high','projected_state_low'} and all(v==0 for v in mism.values()),'trace-on/off batch parity mismatch')
 pts=[json.loads(s) for s in points.read_text().splitlines() if s.strip()]
 rows=[p for p in pts if p.get('record')=='point']
 need(len(rows)==1,'expected exactly one point result row')
 point=rows[0];need((point.get('environment'),point.get('tet_id'),point.get('material_index'))==(0,674,1),'point output row identity mismatch')
 bits=point.get('output_f32_bits');need(isinstance(bits,list) and len(bits)==65,'point output FP32 bit array shape mismatch')
 out=floats_from_bits(bits)
 need(np.array_equal(out[8:17],out[17:26]),'direct stress and tangent-path stress are not bit-identical')
 for key,slc in [('direct_stress_pa',slice(8,17)),('tangent_path_stress_pa',slice(17,26)),('tangent_pa_per_s',slice(26,35)),('projected_state_high',slice(35,50)),('projected_state_low',slice(50,65))]:
  need(np.array_equal(np.asarray(point[key],dtype=np.float32).ravel(),out[slc].astype(np.float32)),'point JSON and raw FP32 words disagree for '+key)
 ref=json.loads(reference_vectors.read_text());rv=ref['point']
 need(ref.get('source_input_sha256')==EXPECTED_INPUT_SHA,'FP64 reference vectors are not bound to origin input')
 # native_material_point_run_sha256 is provenance for the frozen vector source, not a requirement that future regression outputs be byte-identical.
 need(sha(package)==binding['run_binding']['compiled_package_sha256'],'reference package mismatch')
 # The separate source-117 replay proves native deformation readback; verify all F/Fdot words.
 fr=json.loads(f_readback.read_text())
 fm=next(json.loads(line) for line in f_readback_manifest.read_text().splitlines() if json.loads(line).get('record')=='header')
 need(fr.get('record')=='point' and fr.get('compiled_local_iterations')==desc['local_newton_iterations'],'F readback is not the expected source-117 point record')
 need(fm.get('run_package_sha256')==run['compiled_package_sha256'] and fm.get('medium_source_sha256')==run['source_medium_asset_sha256'],'F readback provenance mismatch')
 need((fm.get('environment'),fm.get('tet_id'),fm.get('material_index'))==(0,674,1) and fm.get('run_compiled_local_iterations')==desc['local_newton_iterations'],'F readback source-117 point identity mismatch')
 need(fm.get('candidate_nodes_sha256')==ident['candidate_nodes_sha256'] and fm.get('accepted_state_sha256')==ident['accepted_history_sha256'],'F readback candidate/history binding mismatch')
 words=fr.get('raw_160_float32_words');need(isinstance(words,list) and len(words)==160,'native F readback raw word layout mismatch')
 hi=f32_words(np.asarray(c['F_high']).ravel());lo=f32_words(np.asarray(c['F_low']).ravel());fd=f32_words(np.asarray(c['Fdot']).ravel())
 need(words[3:12]==hi and words[12:21]==lo and words[21:30]==fd,'native deformation high/low/rate words differ from exact fixture')
 need(rv.get('parameters_from_exact_fp32_input')==np.asarray(c['parameters'],dtype=np.float32).astype(float).tolist(),'reference parameter vector mismatch')
 need(rv.get('accepted_state_from_exact_fp32_input')==np.asarray(c['accepted_state'],dtype=np.float32).astype(float).tolist(),'reference accepted state mismatch')
 need(f32_words(np.asarray(rv['F_high']).ravel())==hi and f32_words(np.asarray(rv['F_low']).ravel())==lo and f32_words(np.asarray(rv['deformation_direction']).ravel())==f32_words(c['deformation_direction']),'reference kinematics mismatch')
 # Decode and enforce unchanged raw residual and scaled-correction gates.
 tr=json.loads(trace.read_text());need(tr.get('record')=='header' and tr.get('point_input_sha256')==EXPECTED_INPUT_SHA,'projection trace input binding mismatch')
 need(tr.get('compiled_local_iterations')==desc['local_newton_iterations'],'trace budget differs from compiled material')
 summary=tr['summary'];need(summary.get('direct_project_valid') is True and summary.get('stress_valid') is True and summary.get('tangent_valid') is True and summary.get('tangent_projection_converged') is True,'trace summary reports invalid result')
 tracebits=tr.get('trace_f32_bits');need(isinstance(tracebits,list),'trace is missing raw float bits')
 vals=floats_from_bits(tracebits);layout=tr.get('trace_layout',{});head=int(layout.get('header_slots',0));stride=int(layout.get('iteration_slots',0));count=int(layout.get('max_iterations',0));slots=int(layout.get('slots',0))
 need(head==8 and stride==374 and count==desc['local_newton_iterations'] and len(vals)==slots,'trace layout is unsupported')
 iters=int(summary.get('direct_iterations',0));need(0<iters<=count,'projection iteration count outside compiled bound')
 final=head+(iters-1)*stride
 status=vals[final];raw=float(vals[final+3]);state_norm=float(vals[final+4]);corr=float(vals[final+5])
 need(status==9.0,'final projection iteration did not report converged status')
 raw_limit=2e-8*(1+state_norm)
 need(raw<=raw_limit,'final raw residual gate exceeds unchanged 2e-8*(1+state norm) limit')
 need(corr<=2e-8,'final scaled-correction gate exceeds unchanged 2e-8 limit')
 # Intermediate records must use the frozen dimensionless row merit and accept true decreases.
 row_scale_error=0.0; line_searches=0; line_rows=[]
 accepted=floats_from_bits(c['accepted_state_f32_bits']).astype(np.float32); params=np.asarray(c['parameters'],dtype=np.float32);scale_floor=np.float32(params[26]);statecap=16;nstate=15
 for k in range(iters):
  b=head+k*stride;s=int(vals[b])
  if s==9: continue
  need(s==7,'intermediate projection iteration was not accepted')
  bt=int(vals[b+1]);alpha=float(vals[b+2]);need(0<=bt<16 and alpha>0,'invalid accepted line-search slot')
  state=np.asarray(vals[b+6:b+6+nstate],dtype=np.float32)
  residual=np.asarray(vals[b+6+2*statecap:b+6+2*statecap+nstate],dtype=np.float32)
  J=np.asarray(vals[b+6+5*statecap:b+6+5*statecap+nstate*nstate],dtype=np.float32).reshape(nstate,nstate)
  base=b+6+5*statecap+nstate*nstate
  stored=np.asarray(vals[base:base+nstate],dtype=np.float32)
  scales=np.maximum(np.maximum(np.abs(accepted),np.abs(state)),scale_floor).astype(np.float32)
  calculated=np.zeros(nstate,dtype=np.float32)
  for rr in range(nstate):
   total=np.float32(0)
   for cc in range(nstate):total=np.float32(total+np.float32(abs(J[rr,cc])*scales[cc]))
   calculated[rr]=total
  row_scale_error=max(row_scale_error,float(np.max(np.abs(calculated-stored))))
  need(np.max(np.abs(calculated-stored))<=1e-8,'trace row scales do not match frozen Jacobian/state scaling')
  current=float(np.max(np.abs(residual)/stored))
  trials=np.asarray(vals[base+nstate:base+nstate+16],dtype=float)
  need(bt<len(trials) and math.isfinite(trials[bt]) and trials[bt]>=0,'accepted trial scaled merit is missing')
  trial=float(trials[bt]);need(trial<current and trial<=(1-1e-4*alpha)*current,'accepted trial does not satisfy strict frozen-row Armijo decrease')
  line_searches+=1;line_rows.append({'iteration':k+1,'alpha':alpha,'backtrack':bt,'current_merit':current,'trial_merit':trial})
 # Compare native projected state/stress/tangent to fixed independent FP64 vectors.
 state=out[35:50]+out[50:65];stress=out[8:17];tangent=out[26:35]
 state_m=vector_metric(state,rv['projected_state']);stress_m=vector_metric(stress,rv['stress_Pa']);tangent_m=vector_metric(tangent,rv['tangent_Pa_per_s'])
 for label,m in [('state',state_m),('stress',stress_m),('tangent',tangent_m)]:
  need(m['relative_l2']<=1e-3,label+' relative-L2 difference exceeds 1e-3')
  need(m['max_relative_component_nonzero'] is not None and m['max_relative_component_nonzero']<=1e-3,label+' componentwise relative difference exceeds 1e-3')
 return {'schema':'numi.cardboard.source380-point-regression-verifier-result.v1','status':'pass','point':'environment 0 / tetrahedron 674 / medium bilinear15','compiled_budget':count,'projection_iterations':iters,'final_raw_residual_inf':raw,'final_raw_gate_limit':raw_limit,'final_scaled_correction_inf':corr,'scaled_correction_gate_limit':2e-8,'accepted_armijo_trials':line_searches,'line_searches':line_rows,'row_scale_reconstruction_max_abs':row_scale_error,'fp64_vector_metrics':{'state':state_m,'stress':stress_m,'tangent_one_declared_direction':tangent_m},'trace_on_off_bit_mismatches':mism,'physical_steps':0,'physical_validation':False,'qualification_claim':False}
def main():
 ap=argparse.ArgumentParser()
 for n in ['fixture','package','medium-asset','reference-vectors','receipt','result','points','trace','f-readback','f-readback-manifest']:
  ap.add_argument('--'+n,required=True)
 a=ap.parse_args()
 try:
  print(json.dumps(verify(fixture=a.fixture,package=a.package,medium_asset=a.medium_asset,reference_vectors=a.reference_vectors,receipt=a.receipt,result=a.result,points=a.points,trace=a.trace,f_readback=a.f_readback,f_readback_manifest=a.f_readback_manifest),indent=2))
 except Exception as e:
  print(json.dumps({'status':'reject','reason':str(e)},indent=2));return 2
 return 0
if __name__=='__main__':sys.exit(main())
