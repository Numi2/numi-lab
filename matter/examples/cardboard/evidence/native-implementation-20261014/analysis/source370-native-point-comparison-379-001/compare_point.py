#!/usr/bin/env python3
"""Independent source-bound adjudication of native source-370 point replay 379."""
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, struct, sys
import numpy as np

BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
RUN=BASE/'solver/cached-true-residual-certification-coupon-370'
NATIVE=BASE/'native/exact-source370-material-point-379-002'
OUT=pathlib.Path(__file__).resolve().parent
CPU=BASE/'analysis/source105-failure370-exact-tet1820-001'
REF=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
POL=BASE/'analysis/local-projection-scaled-correction-082/replay_scaled_policy.py'
REPLAY=CPU/'replay.json'


def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def bits_f32(x): return f'0x{struct.unpack("<I",struct.pack("<f",float(np.float32(x))))[0]:08x}'
def words_to_f64(words):
    u=np.asarray([int(x,16) for x in words],dtype='<u4')
    return u.view('<f4').astype(np.float64)
def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    assert spec and spec.loader; sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

ref=import_file('bilinear15_reference',REF)
pol=import_file('replay_scaled_policy',POL)
replay_mod=import_file('replay_tet1820_for_compare',CPU/'replay_tet1820.py')
replay=json.loads(REPLAY.read_text())
header, budget_rows=[json.loads(x) for x in (NATIVE/'output/point-replay.jsonl').read_text().splitlines() if x.strip()]
assert header['record']=='header' and budget_rows['record']=='end'
assert header['schema']=='numi.cardboard.source370-native-material-point.v1'
assert header['physical_steps']==0 and header['physical_validation'] is False
assert budget_rows['status']=='diagnostic_complete'

expected={
 'run_package_sha256':'3c25f41c6c26654ccf3f8b4f4b510bbb64ab0ea0f52b94d05d8b1fa4060fafe5',
 'candidate_nodes_sha256':'1ec2da4c997fac19986f5ad196c01dc653457297ae5da767f891d0bd05ea4b92',
 'accepted_state_sha256':'6f1b2031dc933cacec54142e9521e914d30a5f8f2889d6bd589e46878c3bfbbc',
 'run_manifest_sha256':'67a766eebc35ebfeb26b122899ff9137ae62fb4f0122465cb715e04e8bd1e711',
 'medium_source_sha256':'76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9',
}
for k,v in expected.items(): assert header[k]==v,(k,header[k],v)
for k,v in expected.items():
    filemap={'run_package_sha256':RUN/'output/compiled.nmatterpack',
             'candidate_nodes_sha256':RUN/'output/rejected-conditioning-000002/candidate_nodes.bin',
             'accepted_state_sha256':RUN/'output/material_state_000002_bent.json',
             'run_manifest_sha256':RUN/'output/manifest.json',
             'medium_source_sha256':BASE/'snapshots/source-exact-material-point-112/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'}
    assert sha(filemap[k])==v,(k,sha(filemap[k]),v)
assert json.loads((NATIVE/'receipt.json').read_text())['exit_code']==0
assert json.loads((NATIVE/'receipt.json').read_text())['source_inputs_unchanged'] is True

src_f=replay['deformation_words_source_transcription']
expected_hi=[w for row in src_f['F_high_bits'] for w in row]
expected_lo=[w for row in src_f['F_low_bits'] for w in row]
p=np.asarray(replay['compiled_material']['parameters_fp32'],dtype=np.float32).astype(np.float64)
old=np.asarray(replay['compiled_material']['accepted_state_rolled_back_fp32'],dtype=np.float32).astype(np.float64)
threshold=2.0e-8
rows={}
for budget in (16,32):
    path=NATIVE/f'output/budget{budget}.json'
    d=json.loads(path.read_text())
    assert d['compiled_local_iterations']==budget
    assert d['immutable_run_package_medium_iterations']==16
    assert d['physical_steps']==0 and d['physical_validation'] is False
    words=d['raw_160_float32_words']
    native_hi=words[3:12]; native_lo=words[12:21]
    hi_match=native_hi==expected_hi; lo_match=native_lo==expected_lo
    Fhi=words_to_f64(native_hi).reshape(3,3)
    Flo=words_to_f64(native_lo).reshape(3,3)
    F=Fhi+Flo
    zhi=words_to_f64(words[56:71])
    zlo=words_to_f64(words[88:103])
    z=zhi+zlo
    residual=ref.residual(p,F,old,z)
    residual_scaled=10.0*float(np.max(np.abs(residual)))
    zinf=float(np.max(np.abs(z)))
    residual_threshold=2.0e-8*(1.0+zinf)
    J=ref.jacobian(p,F,old,z)
    delta, correction, xscale, rowscale=pol.solve_scaled(J,residual,old,z,float(p[26]))
    correction_components=np.abs(delta/xscale)
    correction_index=int(np.argmax(correction_components))
    correction_threshold=2.0e-8
    cpu_budget=replay['fp64_independent_scaled_projection']['budgets'][str(budget)]
    cpu_state=np.asarray(cpu_budget['returned_state'],dtype=np.float64)
    state_diff=z-cpu_state
    state_scale=np.maximum(np.maximum(np.abs(old),np.abs(z)),float(p[26]))
    state_scaled_diff=float(np.max(np.abs(state_diff)/state_scale))
    cpu_det=float(replay['failed_record']['native_status_diagnostic_x'])
    directstress=np.asarray(d['material_point_stress_world'],dtype=np.float64).reshape(3,3)
    tangentstress=np.asarray(d['tangent_path_stress_world'],dtype=np.float64).reshape(3,3)
    tangentaction=np.asarray(d['material_tangent_world_for_declared_unit_direction'],dtype=np.float64).reshape(3,3)
    # CPU reference stress/action are in material coordinates. Transform back
    # through the same captured material-frame quaternion convention.
    q=np.asarray(replay['compiled_tet']['material_frame_quaternion_xyzw_fp32'],dtype=np.float32)
    # Source-derived float32 frame transcription from the archived exact-input replay.
    R=np.asarray(replay_mod.frame_from_quaternion(q),dtype=np.float64)
    ref_local_stress=np.asarray(ref.stress(p,F,z),dtype=np.float64)
    ref_world_stress=ref_local_stress@R.T
    direction=np.zeros((3,3),dtype=np.float64); direction[0,0]=1.0
    ref_world_action=np.asarray(ref.tangent(p,F,old,z,direction),dtype=np.float64)@R.T
    stress_abs=(float(np.max(np.abs(directstress-ref_world_stress)))
                if d['direct_stress_valid'] else None)
    stress_rel=(float(np.linalg.norm(directstress-ref_world_stress)/max(np.linalg.norm(ref_world_stress),1e-300))
                if d['direct_stress_valid'] else None)
    action_abs=(float(np.max(np.abs(tangentaction-ref_world_action)))
                if d['tangent_stress_valid'] else None)
    action_rel=(float(np.linalg.norm(tangentaction-ref_world_action)/max(np.linalg.norm(ref_world_action),1e-300))
                if d['tangent_stress_valid'] else None)
    rows[str(budget)]={
      'native':{
        'direct_stress_valid':d['direct_stress_valid'],'tangent_stress_valid':d['tangent_stress_valid'],
        'projection_converged':d['projection_converged'],'projection_flags':d['projection_flags'],
        'terminal_failure':d['terminal_failure'],'local_updates':d['local_updates'],
        'reported_residual_inf':d['projection_residual'],'reported_state_inf':d['projection_state_norm'],
        'determinant':d['determinant']},
      'deformation_word_match':{'F_high_exact_fp32_bits':hi_match,'F_low_exact_fp32_bits':lo_match,
                                'native_F_high_bits':native_hi,'cpu_transcribed_F_high_bits':expected_hi,
                                'native_F_low_bits':native_lo,'cpu_transcribed_F_low_bits':expected_lo,
                                'determinant_difference_vs_original_status_x':d['determinant']-cpu_det},
      'independent_fp64_metrics_at_native_returned_candidate':{
        'residual_inf_scaled_10x':residual_scaled,'residual_gate_threshold':residual_threshold,
        'residual_gate_pass':residual_scaled<=residual_threshold,
        'difference_from_native_reported_residual':residual_scaled-float(d['projection_residual']),
        'scaled_correction_inf':correction,'scaled_correction_gate_threshold':correction_threshold,
        'scaled_correction_gate_pass':correction<=correction_threshold,
        'largest_scaled_correction_state_index':correction_index,
        'largest_scaled_correction_component':float(correction_components[correction_index]),
        'state_inf':zinf},
      'comparison_to_saved_fp64_replay_same_budget':{
        'saved_fp64_status':cpu_budget['status'],'saved_fp64_updates':cpu_budget['iterations'],
        'saved_fp64_correction_inf':cpu_budget['scaled_correction_inf'],
        'saved_fp64_residual_inf_scaled':cpu_budget['residual_inf_scaled'],
        'max_abs_state_difference':float(np.max(np.abs(state_diff)),),
        'max_scaled_state_difference':state_scaled_diff},
      'independent_stress_tangent_comparison':{
        'stress_reference_source':'independent bilinear15_reference.py evaluated at exact widened native F pair and returned native state; source-transcribed float32 material frame rotated to world',
        'stress_absolute_max_component_error':stress_abs,'stress_relative_l2_error':stress_rel,
        'stress_comparison_status':('comparable_diagnostic_only' if d['direct_stress_valid'] else 'not_comparable_native_projection_invalid'),
        'tangent_reference_direction':'local F[0,0]=1; Fdot-direction is 1/dt in native call, but the authored constitutive equations are rate-independent',
        'tangent_absolute_max_component_error':action_abs,'tangent_relative_l2_error':action_rel,
        'tangent_comparison_status':('comparable_diagnostic_only' if d['tangent_stress_valid'] else 'not_comparable_native_projection_invalid'),
        'comparison_is_accuracy_qualification':False}}

report={
 'schema':'numi.cardboard.source370.native-point-comparison.v1',
 'classification':'source-bound single material-point adjudication; not full trajectory or material/coupon qualification',
 'native_run':'native/exact-source370-material-point-379-002',
 'native_receipt_sha256':sha(NATIVE/'receipt.json'),
 'native_command_binary_sha256':sha(NATIVE/'inputs/runtime/matter/numi-matter-cardboard-source370-material-point-check'),
 'native_metallib_sha256':sha(NATIVE/'inputs/runtime/matter/shaders/cardboard_paper_check.metallib'),
 'native_header_sha256':sha(NATIVE/'output/point-replay.jsonl'),
 'native_budget16_sha256':sha(NATIVE/'output/budget16.json'),
 'native_budget32_sha256':sha(NATIVE/'output/budget32.json'),
 'native_source_manifest_sha256':sha(NATIVE/'inputs/external/source370-manifest.json'),
 'native_source_build_binding_sha256':sha(NATIVE/'inputs/external/source-build.json'),
 'native_compiled_package_sha256':sha(NATIVE/'inputs/external/compiled.nmatterpack'),
 'native_material_state_input_sha256':sha(NATIVE/'inputs/external/material_state_000002_bent.json'),
 'native_candidate_node_input_sha256':sha(NATIVE/'inputs/external/candidate_nodes.bin'),
 'native_conditioning_manifest_sha256':sha(NATIVE/'inputs/external/conditioning-manifest.json'),
 'cpu_replay_json_sha256':sha(REPLAY),
 'cpu_replay_script_sha256':sha(CPU/'replay_tet1820.py'),
 'cpu_reference_sha256':sha(REF),
 'scaled_policy_script_sha256':sha(POL),
 'source370_input_hashes':expected,
 'selected_native_point':{'environment':0,'tetrahedron':1820,'material_index':1,'time_step_s':1.0e-4,
                          'original_failed_control_step':1,'original_failure_status':'NM_STATUS_INVALID_DEFORMATION'},
 'original_failure':replay['failed_record'],
 'result':rows,
 'findings':[
   'The native high and low deformation words are compared bit-for-bit to the CPU source transcription; whether true, the CPU transcription itself is not a native evaluation.',
   'At the exact point, budget16 reports iteration-exhausted and invalid stress/tangent; budget32 projects successfully in its point-only private compile. The native point reproduces the local outcome split.',
   'The native checker exports the authored residual infinity norm and state infinity norm, but not the native scaled Newton correction. The correction values in this report are independent FP64 next-Newton corrections computed at the widened native returned states and are labeled as derived, not native certificates.',
   'Stress/tangent comparisons use an independent FP64 constitutive reference and source-transcribed frame transform; they are diagnostic comparisons and do not establish tangent accuracy or calibration.',
   'Budget32 is not a replay of source-370 trajectory because the immutable run package was compiled at16.'
 ],
 'limitations':['No physical Matter transaction or full coupon run was executed by this comparison.','The independent reference uses Python-transcribed equations and is not production Metal arithmetic.','The point checker omits native projectionIterationTrace, so exact native per-iteration scaled-correction and line-search alpha are unavailable.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:rows['16']['native'] if k=='budget16' else rows['32']['native'] for k in ('budget16','budget32')},indent=2))
for b in ('16','32'):
 r=rows[b]
 print('budget',b,'Fbits',r['deformation_word_match']['F_high_exact_fp32_bits'],r['deformation_word_match']['F_low_exact_fp32_bits'],
       'resid',r['independent_fp64_metrics_at_native_returned_candidate']['residual_inf_scaled_10x'],
       'correction',r['independent_fp64_metrics_at_native_returned_candidate']['scaled_correction_inf'],
       'stressRel',r['independent_stress_tangent_comparison']['stress_relative_l2_error'],
       'actionRel',r['independent_stress_tangent_comparison']['tangent_relative_l2_error'])
