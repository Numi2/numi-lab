#!/usr/bin/env python3
"""Base-state residual Cauchy directions for exact native FEM capture 419."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, platform
from pathlib import Path
import numpy as np
from scipy import sparse

BASE = Path('/Users/home/cardboard-implementation-20261014')
CAPTURE = BASE / 'solver/two-stage-first-failure-rich-fd-419/operator-capture'
RUN = CAPTURE.parent
SNAPSHOT = BASE / 'snapshots/source-status5-vtccd-diagnostics-125'
MODULE = SNAPSHOT / 'matter/tools/cardboard_cached_operator_audit.py'
REPLAY = BASE / 'analysis/native-operator-term-replay-result-422'
SWEEP = BASE / 'analysis/native-residual-fd-sweep-result-424'
SWEEP_AUDIT = BASE / 'analysis/native424-residual-sweep-audit-001/report.json'
POSTLINEAR = BASE / 'analysis/captured-residual-cauchy-419-001'
OUT = Path(__file__).resolve().parent


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def dot64(a,b):
    aa=np.asarray(a,dtype=np.float64).reshape(-1); bb=np.asarray(b,dtype=np.float64).reshape(-1)
    if aa.shape!=bb.shape: raise ValueError('dot shape mismatch')
    return math.fsum(float(x)*float(y) for x,y in zip(aa,bb))


def norm(x): return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))


def load_module():
    spec=importlib.util.spec_from_file_location('source125_cached_operator_audit',MODULE)
    if spec is None or spec.loader is None: raise RuntimeError('cannot load bound source125 assembly module')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def strict_capture(path):
    def parse_int(s): return -0.0 if s=='-0' else int(s)
    h=integ=None;nodes=[];tets=[]
    with path.open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            row=json.loads(line,parse_int=parse_int,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
            kind=row.get('record')
            if kind=='header':
                if h is not None: raise ValueError('duplicate capture header')
                h=row
            elif kind=='node': nodes.append(row)
            elif kind=='tetrahedron': tets.append(row)
            elif kind=='integrity':
                if integ is not None: raise ValueError('duplicate integrity record')
                integ=row
            else: raise ValueError(f'unknown capture record at line {line_no}')
    if h is None or integ is None: raise ValueError('capture lacks header or integrity record')
    return h,nodes,tets,integ


def write_bin(path,array,dtype):
    a=np.asarray(array,dtype=np.dtype(dtype).newbyteorder('<'));a.tofile(path)
    return {'path':path.name,'dtype':a.dtype.str,'shape':list(a.shape),'bytes':path.stat().st_size,'sha256':sha256(path)}


def direction_record(label, p, A, base_residual, gradient, dt, positions_f32, free_nodes, delta_norm):
    p=np.asarray(p,dtype=np.float64)
    trial=base_residual-A@p
    phi0=.5*dot64(base_residual,base_residual);phi1=.5*dot64(trial,trial)
    displacement=dt*p.reshape(-1,3)
    fullpos=positions_f32[free_nodes].astype(np.float64)+displacement
    rounded=fullpos.astype(np.float32)
    orig=positions_f32[free_nodes]
    changed=np.any(rounded.view(np.uint32)!=orig.view(np.uint32),axis=1)
    return {
      'semantics':'Alternative total correction direction from the captured Newton center; not an increment added to captured delta_v.',
      'step_l2_m_per_s':norm(p),'step_linf_m_per_s':float(np.max(np.abs(p))),
      'step_over_captured_newton_delta_norm':norm(p)/delta_norm,
      'position_increment_linf_m':float(np.max(np.abs(displacement))),
      'estimated_free_nodes_with_any_position_word_changed':int(np.count_nonzero(changed)),
      'predicted_base_residual_l2_Ns':norm(trial),
      'predicted_base_residual_ratio':norm(trial)/norm(base_residual),
      'merit_phi_before_N2s2':phi0,'merit_phi_after_N2s2':phi1,
      'predicted_merit_fractional_drop':(phi0-phi1)/phi0,
      'base_merit_directional_derivative':-dot64(gradient,p),
      'position_rounding_note':'Estimate only: add dt*p in FP64 to captured center positions promoted from FP32, then round once to FP32. This is not the production accepted/current candidate composition order.'
    }


def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=OUT,help='fresh output directory; never overwrites report or manifest')
    OUT=ap.parse_args().output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'report.json').exists() or (OUT/'manifest.json').exists(): raise SystemExit('refusing to overwrite existing result')
    coa=load_module();prediction=RUN/'prediction.json';receipt=RUN/'receipt.json';stderr=RUN/'stderr.log'
    binding=coa.bind_inputs(CAPTURE,prediction,receipt,stderr,0,source_snapshot_override=SNAPSHOT)
    header,nodes,tets,integrity=strict_capture(CAPTURE)
    if (header.get('root'),header.get('newton_iteration'))!=(1,48): raise ValueError('wrong capture selection')
    mass,stiff_scaled,A,delta,meta=coa.validate_and_assemble(header,nodes,tets,integrity,0)
    if A.shape!=(2538,2538): raise ValueError(f'unexpected free operator shape {A.shape}')
    nnode=int(meta['node_count']); free=np.asarray(meta['free_nodes'],dtype=np.int64); dt=float(meta['dt_seconds'])
    b=np.asarray(meta['native_rhs_free_impulse_Ns'],dtype=np.float64)
    Ad=np.asarray(A@delta).reshape(-1); post_r=b-Ad
    g=np.asarray(A.T@b).reshape(-1)
    gpost=np.asarray(A.T@post_r).reshape(-1)
    colnorm=np.sqrt(np.asarray(A.power(2).sum(axis=0)).reshape(-1))
    if not np.isfinite(colnorm).all() or np.any(colnorm<=0): raise ValueError('invalid operator column scale')
    S=1.0/colnorm

    # Unscaled Cauchy in velocity variables.
    Ag=np.asarray(A@g).reshape(-1); den=dot64(Ag,Ag)
    if not np.isfinite(den) or den<=0: raise ValueError('raw base-state Cauchy denominator invalid')
    raw_alpha=dot64(g,g)/den; p_raw=raw_alpha*g
    # Column-equilibrated Cauchy: B=A*S, q=B^T*b=S*A^T*b, p=S*z.
    q=S*g; Bq=np.asarray(A@(S*q)).reshape(-1); den_s=dot64(Bq,Bq)
    if not np.isfinite(den_s) or den_s<=0: raise ValueError('scaled base-state Cauchy denominator invalid')
    alpha_s=dot64(q,q)/den_s; p_scaled=S*(alpha_s*q)

    delta_norm=norm(delta); positions=np.asarray(meta['node_positions_m'],dtype=np.float64).astype(np.float32)
    base_dirs={
       'base_raw_cauchy':p_raw,
       'base_column_equilibrated_cauchy':p_scaled,
       'base_column_equilibrated_cauchy_025':.25*p_scaled,
       'base_column_equilibrated_cauchy_050':.50*p_scaled,
    }
    # A transparent radius family, capped in Euclidean velocity-correction norm.
    for frac in (1e-6,1e-5,1e-4):
       radius=frac*delta_norm; factor=min(1.0,radius/norm(p_scaled))
       base_dirs[f'base_column_equilibrated_trust_{frac:.0e}_delta']=factor*p_scaled
    # Explicitly evaluate the alternative semantics if the Cauchy vector were
    # added to, rather than substituted for, the captured Newton correction.
    # This is a candidate direction delta+p, not the primary fallback.
    total_newton_plus_base_cauchy=delta+p_scaled
    total_full=np.zeros((nnode,4),dtype=np.float64)
    total_full[free,:3]=total_newton_plus_base_cauchy.reshape(-1,3)
    total_displacement=dt*total_full[:,:3]
    total_position=(positions.astype(np.float64)+total_displacement).astype(np.float32)
    total_changed=np.any(total_position.view(np.uint32)!=positions.view(np.uint32),axis=1)

    dirs=[]; stats={}
    for name,p in base_dirs.items():
       stats[name]=direction_record(name,p,A,b,g,dt,positions,free,delta_norm)
       f=np.zeros((nnode,4),dtype=np.float64);f[free,:3]=p.reshape(-1,3)
       dirs.append(write_bin(OUT/f'{name}.full-node-delta-v.float32x4.bin',f,'<f4'))
       dirs.append(write_bin(OUT/f'{name}.free-delta-v.float64le.bin',p,'<f8'))
    total_residual=b-A@total_newton_plus_base_cauchy
    phi0=.5*dot64(b,b);phi_total=.5*dot64(total_residual,total_residual)
    # The old package's post-linear correction is retained as its own distinct
    # residual-gradient experiment: p_post is generated from b-A*delta.
    gpostA=np.asarray(A.T@post_r).reshape(-1); Agpost=np.asarray(A@gpostA).reshape(-1)
    post_alpha=dot64(gpostA,gpostA)/dot64(Agpost,Agpost); p_post=post_alpha*gpostA
    post_total_residual=post_r-A@p_post

    # Matrix-free gather/scatter and transpose identity.
    dofmap=np.full((nnode,3),-1,dtype=np.int64);dofmap[free]=3*np.arange(len(free))[:,None]+np.arange(3)[None,:]
    local_cache=[]; dt2=dt*dt
    tets_env={int(t['tet_id']):t for t in tets if int(t['environment'])==0}
    for tid,tet in tets_env.items():
       tangent=coa.captured_f32_array(tet['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F'],(81,),f'tet{tid} tangent').reshape(9,9)
       mloc=coa.captured_f32_array(tet['center']['dt_seconds_bulk_modulus_kind_frame_active'],(4,),f'tet{tid} metadata')
       inv=coa.captured_f32_array(tet['inverse_rest_rows_per_m'],(3,3),f'tet{tid} inv')
       vol=coa.captured_f32_scalar(tet['rest_volume_m3'],f'tet{tid} vol')
       quat=coa.captured_f32_array(tet['material_frame_quaternion_xyzw'],(4,),f'tet{tid} q')
       ke=coa.local_tet_stiffness(inv,vol,tangent,quat,bool(mloc[3]==1.0))*dt2
       conn=np.asarray(tet['nodes'],dtype=np.int64);ids=np.array([dofmap[int(n),ax] for n in conn for ax in range(3)])
       keep=np.flatnonzero(ids>=0);local_cache.append((ids[keep],ke[np.ix_(keep,keep)]))
    md=mass.diagonal()
    def mf(x,transpose=False):
       y=md*np.asarray(x,dtype=np.float64)
       for ids,ke in local_cache: np.add.at(y,ids,(ke.T if transpose else ke)@x[ids])
       return y
    rng=np.random.default_rng(0x419A);vx=rng.standard_normal(A.shape[0]);vy=rng.standard_normal(A.shape[0])
    mf_err=norm(mf(vx)-A@vx)/norm(A@vx)
    mft_err=norm(mf(vy,True)-A.T@vy)/norm(A.T@vy)
    lhs=dot64(vy,mf(vx));rhs=dot64(vx,mf(vy,True));adjerr=abs(lhs-rhs)/max(abs(lhs),abs(rhs))
    if max(mf_err,mft_err,adjerr)>2e-12: raise ValueError('matrix-free/adjoint check exceeded 2e-12')

    # Native action and previous endpoint sweep are separately bound inputs.
    replay_report_path=REPLAY/'result/replay-report.json'; replay_report=json.loads(replay_report_path.read_text())
    replay_receipt=json.loads((REPLAY/'receipt.json').read_text())
    def clean_native(r):
       s=r.get('commands',{}).get('native',r)
       return s.get('exit_code')==0 and s.get('timed_out') is False
    if not clean_native(replay_receipt): raise ValueError('422 native replay did not complete cleanly')
    if replay_report.get('capture_selection')!={'control_step':1,'newton_iteration':48} or replay_report.get('physical_transactions')!=0:
       raise ValueError('422 replay does not match selected diagnostic capture')
    sweep=json.loads(SWEEP_AUDIT.read_text()); sweep_receipt=json.loads((SWEEP/'receipt.json').read_text())
    if not clean_native(sweep_receipt) or sweep.get('capture_sha256')!=sha256(CAPTURE): raise ValueError('424 sweep completion/capture binding failure')
    postlinear_manifest=json.loads((POSTLINEAR/'manifest.json').read_text())
    if postlinear_manifest.get('capture_sha256')!=sha256(CAPTURE): raise ValueError('prior post-linear experiment is not bound to capture 419')
    cached=np.fromfile(REPLAY/'result/cached-action.float4.bin',dtype='<f4').reshape(2,nnode,4)[0,:,:3].astype(np.float64)[free].reshape(-1)
    replay_r=b-cached
    captured_action=np.asarray(meta['action_component_arrays']['native_Ns'])
    captured_r=b-captured_action
    gcos=lambda a,c:dot64(a,c)/(norm(a)*norm(c))
    input_hashes={
      'capture_sha256':sha256(CAPTURE),'prediction_sha256':sha256(prediction),'receipt_sha256':sha256(receipt),'stderr_sha256':sha256(stderr),
      'source_snapshot':str(SNAPSHOT),'assembly_module_sha256':sha256(MODULE),
      '422_manifest_sha256':sha256(REPLAY/'manifest.json'),'422_report_sha256':sha256(replay_report_path),
      '422_receipt_sha256':sha256(REPLAY/'receipt.json'),'422_cached_action_sha256':sha256(REPLAY/'result/cached-action.float4.bin'),
      '424_manifest_sha256':sha256(SWEEP/'manifest.json'),'424_audit_sha256':sha256(SWEEP_AUDIT),'424_receipt_sha256':sha256(SWEEP/'receipt.json')}
    input_hashes.update({
      'postlinear_package_manifest_sha256':sha256(POSTLINEAR/'manifest.json'),
      'postlinear_report_sha256':sha256(POSTLINEAR/'report.json'),
      'postlinear_interpretation_erratum_sha256':sha256(POSTLINEAR/'ERRATUM.md'),
      'postlinear_column_cauchy_vector_sha256':sha256(POSTLINEAR/'column_equilibrated_cauchy.free-delta-v.float64le.bin'),
    })
    artifacts=[]
    artifacts.append(write_bin(OUT/'Atranspose_base_rhs_gradient.float64le.bin',g,'<f8'))
    artifacts.append(write_bin(OUT/'base_rhs.float64le.bin',b,'<f8'))
    artifacts.append(write_bin(OUT/'captured_newton_delta.float64le.bin',delta,'<f8'))
    artifacts.append(write_bin(OUT/'delta_plus_base_column_equilibrated_cauchy.total-correction.float64le.bin',total_newton_plus_base_cauchy,'<f8'))
    artifacts.append(write_bin(OUT/'delta_plus_base_column_equilibrated_cauchy.total-node-correction.float32x4.bin',total_full,'<f4'))
    artifacts.append(write_bin(OUT/'post_newton_linear_remainder.float64le.bin',post_r,'<f8'))
    artifacts.append(write_bin(OUT/'column_norms.float64le.bin',colnorm,'<f8'))
    matpath=OUT/'A_free_csr.npz';sparse.save_npz(matpath,A)
    artifacts.append({'path':matpath.name,'format':'scipy-csr-npz','shape':list(A.shape),'nnz':int(A.nnz),'bytes':matpath.stat().st_size,'sha256':sha256(matpath)})
    artifacts.extend(dirs)
    report={
      'schema':'numi.cardboard.captured-base-state-cauchy-experiment.v1',
      'verdict':'BASE_STATE_LINEAR_CANDIDATE_READY_FOR_NATIVE_RESIDUAL_REPLAY',
      'scope':'Offline FP64 linearized residual-merit direction from the exact source125 rejected Newton-48 center. The Cauchy vectors are alternative total correction directions from that center; they are not increments added to the captured Newton delta unless explicitly named.',
      'selection':{'root':1,'newton_iteration':48,'environment':0,'world_fingerprint_hex':header.get('world_fingerprint_hex'),'execution_fingerprint_hex':header.get('execution_fingerprint_hex'),'physical_transactions':0},
      'input_bindings':input_hashes,
      'source_input_binding_result':binding,
      'operator':{'assembly':meta['assembly'],'matrix_layout':meta['matrix_layout'],'shape':list(A.shape),'nnz':int(A.nnz),'free_nodes':meta['free_node_count'],'fixed_nodes':meta['fixed_node_count'],'constraint_tag_counts':meta['constraint_tag_counts'],'fixed_correction_max_abs_m_per_s':meta['action_parity']['fixed_correction_max_abs_m_per_s'],'fixed_native_action_max_abs_Ns':meta['action_parity']['fixed_rows_max_abs_native_action_Ns'],'dt_seconds':dt,'relative_skew_norm':meta['relative_skew_norm'],'matrix_free_adjoint_check':{'seed':0x419A,'forward_relative_error':mf_err,'transpose_relative_error':mft_err,'dot_left':lhs,'dot_right':rhs,'adjoint_relative_error':adjerr,'passed':True}},
      'residuals':{
        'base_nonlinear_center_residual_b_l2_Ns':norm(b),
        'captured_correction_delta_v_l2_m_per_s':norm(delta),
        'assembled_A_delta_l2_Ns':norm(Ad),
        'post_newton_linear_remainder_b_minus_A_delta_l2_Ns':norm(post_r),
        '422_native_cached_action_residual_l2_Ns':norm(replay_r),
        'capture_native_action_residual_l2_Ns':norm(captured_r),
        'assembled_vs_422_cached_action_l2_Ns':norm(Ad-cached),
        'assembled_vs_capture_action_l2_Ns':norm(Ad-captured_action),
        '424_endpoint_sweep_all_retained_endpoints_bitwise_equal':sweep.get('all_retained_endpoints_bitwise_equal'),
        '424_derivative_qualified':sweep.get('derivative_qualified'),
        'distinction':'The base fallback uses R=b and gradient A^T b. The previous package used R=b-A*delta and is a post-Newton linear-residual refinement, not a fallback direction from this center.'},
      'base_state_gradient':{'formula':'g=A^T b for phi(v)=1/2||b-A v||^2 at v=0 in this Newton-center coordinate','l2':norm(g),'linf':float(np.max(np.abs(g))),'column_equilibration':'S_jj=1/||A[:,j]||_2; B=A*S; q=B^T*b; p=S*((q^T q)/(||Bq||_2^2))*q'},
      'base_state_directions':stats,
      'total_newton_plus_base_cauchy':{
        'semantics':'δ+p, explicitly additive alternative only; primary fallback remains p substituted for δ.',
        'delta_plus_p_l2_m_per_s':norm(total_newton_plus_base_cauchy),
        'position_increment_linf_m':float(np.max(np.abs(total_displacement))),
        'estimated_free_nodes_with_any_position_word_changed':int(np.count_nonzero(total_changed[free])),
        'predicted_residual_l2_Ns':norm(total_residual),
        'predicted_residual_ratio_to_base_b':norm(total_residual)/norm(b),
        'predicted_merit_fractional_drop_from_base':(phi0-phi_total)/phi0,
      },
      'post_newton_refinement_supplement':{
        'residual_definition':'r=b-Aδ', 'gradient_l2':norm(gpostA),
        'cauchy_increment_l2_m_per_s':norm(p_post),
        'predicted_remainder_after_increment_l2_Ns':norm(post_total_residual),
        'classification':'Supplementary iterative-residual-refinement direction; not the current-center fallback.'},
      'limitations':['Capture is a rejected Newton candidate and does not represent an accepted physical state.','This operator is the FEM-only block for this compiled fixture; it has zero rigid proxies, contacts, fields, and cohesive faces.','Linear model predictions are not nonlinear residual evaluations or line-search acceptance. Root must test the saved primary p directions at the same center through the production residual replay.','Native cached-action and captured-action discrepancies are preserved and are not assigned a numerical noise floor.','The 424 endpoint sweep was bitwise consistent but did not qualify an FD derivative; this analysis does not turn it into a noise-floor estimate.','No physical operator, force, tolerance, or acceptance policy was changed.'],
      'artifacts':artifacts,'python':platform.python_version(),'numpy':np.__version__}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'verdict':report['verdict'],'base_b_l2_Ns':norm(b),'raw':stats['base_raw_cauchy'],'column_scaled':stats['base_column_equilibrated_cauchy'],'delta_plus_p':report['total_newton_plus_base_cauchy'],'report':str(OUT/'report.json')},indent=2,allow_nan=False))

if __name__=='__main__': main()
