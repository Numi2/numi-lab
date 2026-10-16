#!/usr/bin/env python3
"""Inspect a sealed bilinear kraft/NSSC capture through independent FP64 layers.

This reader is diagnostic only. It reprojects the captured center and every
realized endpoint from the same accepted history; it never advances Matter or
promotes a local derivative comparison into a nonlinear/convergence verdict.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import cardboard_bilinear_fp64_reference as ref
import cardboard_cached_operator_audit as geometry

PAPER_NAMES = (
    'density','c11','c22','c33','c12','c13','c23','g12','g13','g23',
    'sigma_ref','hill_F','hill_G','hill_H','hill_L','hill_M','hill_N',
    'hardening_md','hardening_cd','fit_transfer_scale_md','fit_transfer_scale_cd',
    'hardening_md_flow_scale','hardening_cd_flow_scale','hardening_smoothing_width',
    'hardening_weight_scale','q_floor','fb_epsilon','residual_scale')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def f32(value):
    a=np.asarray(value,dtype=np.float32).astype(np.float64)
    if not np.isfinite(a).all(): raise ValueError('nonfinite captured FP32 word')
    return a

def pair(record,key):
    high=f32(record[key]);low=f32(record.get(key+'_low',np.zeros_like(high)))
    if high.shape!=low.shape: raise ValueError('high/low word shape mismatch')
    return high+low

def comparison(actual,expected,gate=None):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    if a.shape!=b.shape or a.ndim<2: raise ValueError('comparison shape mismatch')
    a,b=a.reshape(len(a),int(np.prod(a.shape[1:]))),b.reshape(len(b),int(np.prod(b.shape[1:])))
    valid=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
    error=np.linalg.norm(a-b,axis=1);scale=np.linalg.norm(b,axis=1)
    relative=np.full(len(a),np.nan)
    np.divide(error,scale,out=relative,where=valid&(scale>0))
    resolved=valid&(scale>0)
    report={'rows':len(a),'finite_rows':int(valid.sum()),
            'zero_reference_rows':int(np.sum(valid&(scale==0))),
            'max_absolute_l2':float(error[valid].max()) if valid.any() else None,
            'max_reference_l2':float(scale[valid].max()) if valid.any() else None,
            'max_relative_l2':float(relative[resolved].max()) if resolved.any() else None,
            'relative_gate':gate,
            'all_rows_at_relative_gate':bool(resolved.all() and np.all(relative<=gate)) if gate else None,
            'near_zero_policy':'Zero-reference or failed rows stay inconclusive; no measured native absolute noise floor is inferred from FP64 arithmetic.'}
    return report,relative,error,scale

def finite_report(value, path='$'):
    """Keep every unresolved scalar visible while writing strict JSON.

    Raw numerical arrays retain NaN/Inf in the NPZ. JSON uses null plus an
    explicit location/value ledger, so a failed reference projection cannot
    silently disappear or prevent the remaining diagnostic from being saved.
    """
    if isinstance(value,float) and not math.isfinite(value):
        return None,[{'path':path,'value':repr(value),'classification':'unresolved_nonfinite_diagnostic'}]
    if isinstance(value,dict):
        result={};missing=[]
        for key,item in value.items():
            child_path=path+'/'+str(key).replace('~','~0').replace('/','~1')
            result[key],rows=finite_report(item,child_path);missing.extend(rows)
        return result,missing
    if isinstance(value,(list,tuple)):
        result=[];missing=[]
        for index,item in enumerate(value):
            clean,rows=finite_report(item,path+'/'+str(index));result.append(clean);missing.extend(rows)
        return result,missing
    return value,[]

def project(p,F,old):
    seed=ref.project_centers(ref,p,F,old,p[:,26])
    return ref.continue_projection_tight(ref,p,F,old,p[:,26],seed)

def branch_values(p,old,z):
    b=ref.paper_branch(ref,p,old,z)
    dl=b['plastic_multiplier_increment'];y=b['yield_function_residual_value']
    den=np.sqrt(dl*dl+y*y+p[:,26]**2)+dl+y
    return np.stack((dl,y,den),axis=1)

def branch(p,old,z):
    dl,y,den=branch_values(p,old,z).T
    return np.stack((np.sign(dl),np.sign(y),den>1e-14),axis=1)

def forces(stress,inverse,volume):
    h=-volume[:,None,None]*stress@np.swapaxes(inverse,-1,-2)
    out=np.empty((len(stress),4,3));out[:,1:]=np.swapaxes(h,-1,-2)
    out[:,0]=-out[:,1:].sum(axis=1)
    return out

def gather(element,tet_nodes,node_count):
    out=np.zeros((node_count,3))
    for slot in range(4): np.add.at(out,tet_nodes[:,slot],element[:,slot])
    return out

def validate_identity(header,nodes,tets,bindings):
    envs=header.get('environment_count');nn=header.get('node_count_per_environment');nt=header.get('tetrahedron_count_per_environment')
    if type(envs) is not int or envs!=2 or type(nn) is not int or not 0<nn<=2048 or type(nt) is not int or not 0<nt<=8192:
        raise ValueError('bounded two-environment capture counts required')
    for records,count,id_key in ((nodes,nn,'node_id'),(tets,nt,'tet_id')):
        identity=[(r['environment'],r[id_key]) for r in records]
        if any(type(e) is not int or type(i) is not int for e,i in identity) or len(identity)!=envs*count or set(identity)!={(e,i) for e in range(envs) for i in range(count)}:
            raise ValueError('incomplete/duplicate/out-of-range capture identity')
    root,iteration=map(int,bindings['prediction_selector'].split(':'))
    if (header.get('root'),header.get('newton_iteration'))!=(root,iteration): raise ValueError('capture selection differs from sealed run')
    diagnostic=bindings['operator_fd_iteration']
    for label in ('world','execution'):
        if int(header[label+'_fingerprint_hex'],16)!=int(diagnostic[label+'_fingerprint']):
            raise ValueError('capture '+label+' fingerprint differs from terminal diagnostic')
    cache=header.get('cached_algorithmic_tangent_capture',{})
    if cache.get('matrix_layout')!='row-major 9x9, local material frame' or 'dP_local/dF_local' not in cache.get('derivative_contract',''):
        raise ValueError('unsupported cached derivative layout')
    return root,iteration

def audit(case,output,environment=0,epsilon_powers=range(8,27,2)):
    started=time.monotonic();case=Path(case).resolve();output=Path(output).resolve()
    epsilon_powers=list(epsilon_powers)
    if not 2<=len(epsilon_powers)<=16 or any(type(p) is not int or not 0<=p<=40 for p in epsilon_powers) or any(a>=b for a,b in zip(epsilon_powers,epsilon_powers[1:])):
        raise ValueError('epsilon grid must have2..16 strictly increasing integer powers in0..40')
    if output.exists(): raise ValueError('fresh output required; preserve earlier attempts')
    pred=json.loads((case/'prediction.json').read_text())
    receipt=json.loads((case/'receipt.json').read_text())
    if receipt.get('execution_started') is not True or receipt.get('source_inputs_unchanged') is not True or receipt.get('changed_inputs')!=[]:
        raise ValueError('native execution/input receipt is incomplete or changed')
    capture=Path(pred['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH']).resolve()
    bindings=geometry.bind_inputs(capture,case/'prediction.json',case/'receipt.json',case/'stderr.log',environment)
    manifest=json.loads((case/'output/manifest.json').read_text())
    solver=manifest['solver']
    if solver.get('maximum_rate_exponent')!=0 or solver.get('deformable_self_contact') is not False or solver.get('gravity_m_s2')!=[0,0,0]:
        raise ValueError('reader supports the rate-independent, zero-gravity, contact-free coupon only')
    for key in ('corrugator_starch_interface','retained_matrix_damage_overlay','shipping_hma_closure'):
        if manifest.get(key,{}).get('enabled') is not False: raise ValueError('unsupported interface/damage law: '+key)
    header,nodes,tets,integrity=geometry.capture_records(capture)
    root,iteration=validate_identity(header,nodes,tets,bindings)
    # This performs complete topology, cache, constraint and frame validation.
    geometry.validate_and_assemble(header,nodes,tets,integrity,environment)
    bindings['capture_run_fingerprints_match']=True
    nodes=sorted((r for r in nodes if r['environment']==environment),key=lambda r:r['node_id'])
    tets=sorted((r for r in tets if r['environment']==environment),key=lambda r:r['tet_id'])
    if len(tets)>8192 or len(nodes)>2048: raise ValueError('bounded CPU diagnostic scope exceeded')
    if any(len(t['sides'])!=4 for t in tets): raise ValueError('both realized epsilon pairs required')
    specs={m['index']:m for m in header['materials']};pt=f32(header['parameters_by_environment'][environment])
    expected={0:'hajali2009_liner_hill_bilinear_fit',1:'hajali2009_medium_hill_bilinear_fit',2:'starch2007_finite_glue_elastic'}
    if set(specs)!=set(expected) or any(specs[i]['name']!=name for i,name in expected.items()): raise ValueError('material identity differs from independent law')
    if any(tuple(specs[i]['parameter_names'])!=PAPER_NAMES for i in (0,1)) or specs[2]['parameter_names']!=['density','mu','lambda']:
        raise ValueError('material parameter ordering differs from independent law')
    material=np.array([t['material_index'] for t in tets]);paper=np.flatnonzero(material!=2);glue=np.flatnonzero(material==2)
    if any(tets[i]['material_state_count']!=15 or tets[i]['material_parameter_count']!=28 for i in paper): raise ValueError('paper state layout differs')
    p=np.stack([pt[specs[material[i]]['parameter_offset']:specs[material[i]]['parameter_offset']+28] for i in paper])
    old=np.stack([f32(tets[i]['center']['accepted_state']) for i in paper])
    if old.shape!=(len(paper),15): raise ValueError('accepted paper history shape differs')
    gi=specs[2]['parameter_offset'];mu=np.full(len(glue),pt[gi+1]);lam=np.full(len(glue),pt[gi+2])
    inverse=np.stack([f32(t['inverse_rest_rows_per_m']) for t in tets]);volume=f32([t['rest_volume_m3'] for t in tets])
    rotation=np.stack([geometry.captured_material_frame(t, t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]==1) for t in tets])
    ids=np.array([t['nodes'] for t in tets],int);dt=float(f32(header['dt_seconds']))
    centerF=np.stack([pair(t['center'],'F_local') for t in tets]);direction=np.stack([f32(t['center']['F_direction_local_per_s']) for t in tets])
    cached=np.stack([f32(t['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']).reshape(9,9) for t in tets])
    output.mkdir(parents=True);arrays={'material':material,'tet_nodes':ids,'F_center':centerF,'accepted_paper_history':old}
    report={'schema':'numi.cardboard.bilinear-capture-layers.v1','classification':'independent FP64 diagnostic at captured candidates; no native transaction or convergence qualification',
            'scope':{'environment':environment,'control':root,'newton':iteration,'tets':len(tets),'nodes':len(nodes),'dt_s':dt},
            'bindings':bindings,'tools_sha256':{Path(x).name:sha(x) for x in (__file__,ref.__file__,geometry.__file__)},'states':[],
            'six_qualification_axes':{k:False for k in ('solver_correctness','numerical_convergence','component_calibration','analogue_validation','physical_assembly','crush_impact_prediction')}}
    center_z=None;center_valid=None;center_derivative=None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for state_index in range(5):
            states=[t['center'] if state_index==0 else t['sides'][state_index-1] for t in tets]
            F=np.stack([pair(s,'F_local') for s in states]);metal_z=np.stack([pair(states[i],'projected_state') for i in paper])
            z,it,rn,cn,valid,reasons,passes=project(p,F[paper],old)
            P=np.full_like(F,np.nan);P[paper[valid]]=ref.stress(p[valid],F[paper[valid]],z[valid]);P[glue]=ref.glue_stress(F[glue],mu,lam)
            metalP=np.stack([f32(s['P_local_Pa']) for s in states])
            native_r=np.max(np.abs(ref.residual(p,F[paper],old,metal_z))*np.array([1e4]*6+[1]*6+[1e4]*3),axis=1)
            state_report={'state_index':state_index,'projection':{'rows':len(paper),'valid_rows':int(valid.sum()),'max_scaled_residual':float(rn[valid].max()) if valid.any() else None,'max_dynamic_correction':float(cn[valid].max()) if valid.any() else None,'max_independent_scaled_residual_at_native_state':float(native_r.max()),'failures':[{'tet':int(paper[j]),'reason':reasons[j]} for j in np.flatnonzero(~valid)],'independent_refinement':passes},
                          'stress_comparison_Pa':comparison(metalP,P)[0], 'projected_state_comparison':comparison(metal_z,z)[0]}
            state_report['projection']['failure_meaning']='Failed independent FP64 reference projections make those rows inconclusive; they do not by themselves demonstrate a production projection defect.'
            native_branch=branch(p,old,metal_z);reference_branch=branch(p,old,z)
            state_report['native_vs_reference_branch_mismatch_rows']=int(np.sum(valid&np.any(native_branch!=reference_branch,axis=1)))
            native_values=branch_values(p,old,metal_z);reference_values=branch_values(p,old,z)
            state_report['branch_interpretation']='Independently evaluated lambda/yield sign and FB-denominator predicates, not captured native internal branch traces. Signs at tiny margins can disagree without a physical active-set change in the regularized law.'
            state_report['native_predicate_absolute_margin_minima']={key:float(np.abs(native_values[:,j]-(1e-14 if j==2 else 0)).min()) for j,key in enumerate(('lambda_increment','yield_residual','fb_denominator_cutoff'))}
            arrays.update({f's{state_index}_F':F,f's{state_index}_reference_state':z,f's{state_index}_reference_stress':P,f's{state_index}_native_state':metal_z,f's{state_index}_native_residual':native_r,f's{state_index}_valid':valid,f's{state_index}_iterations':it,f's{state_index}_scaled_residual':rn,f's{state_index}_dynamic_correction':cn})
            arrays[f's{state_index}_native_branch_values']=native_values;arrays[f's{state_index}_reference_branch_values']=reference_values
            if state_index==0:
                center_z=z;center_valid=valid
                C=np.full_like(cached,np.nan)
                for column in range(9):
                    H=np.zeros_like(F);H.reshape(len(F),9)[:,column]=1
                    C[paper[valid],:,column]=ref.tangent(p[valid],F[paper[valid]],old[valid],z[valid],H[paper[valid]]).reshape(int(valid.sum()),9)
                    C[glue,:,column]=ref.glue_tangent_action(F[glue],H[glue],mu,lam).reshape(len(glue),9)
                state_report['cached_material_matrix_comparison_Pa']=comparison(np.swapaxes(cached,-1,-2).reshape(-1,9),np.swapaxes(C,-1,-2).reshape(-1,9),1e-3)[0]
                analytic=np.einsum('tij,tj->ti',C,direction.reshape(len(F),9)).reshape(len(F),3,3);center_derivative=analytic
                local=np.stack([f32(t['center']['tangent_local_Pa_per_s']) for t in tets]);world=np.stack([f32(t['center']['tangent_world_Pa_per_s']) for t in tets])
                expected_world=analytic@np.swapaxes(rotation,-1,-2)
                native_element=np.stack([f32(t['center']['operator_force_node_world_N_per_s']) for t in tets]);element=forces(expected_world,inverse,volume)
                state_report['local_directional_action']=comparison(local,analytic,1e-3)[0]
                state_report['frame_action']=comparison(world,expected_world,1e-3)[0]
                state_report['element_action']=comparison(native_element,element,1e-3)[0]
                nodal=gather(element,ids,len(nodes));mass=f32([n['center']['mass_kg'] for n in nodes]);dv=np.stack([f32(n['correction_delta_v_m_s']) for n in nodes]);free=~np.isin(f32([n['center']['constraint_tag'] for n in nodes]),[1,3])
                mass_action=mass[:,None]*dv;stiffness_action=-dt*dt*nodal;total=mass_action+stiffness_action
                native=np.stack([f32(n['production_A_delta_v_impulse_Ns'])[:3] for n in nodes])
                state_report['assembled_action_Ns']=comparison(native[free][None],total[free][None],.01)[0]
                state_report['separate_mass_and_stiffness']={'mass_action_l2_Ns':float(np.linalg.norm(mass_action[free])),'stiffness_action_l2_Ns':float(np.linalg.norm(stiffness_action[free])),'sign_convention':'K=-df_internal/dx; A=M+dt^2*K; perturb x by dt*epsilon*delta_v'}
                native_force=np.stack([f32(t['center']['element_force_node_world_N']) for t in tets])
                reference_force=forces(P@np.swapaxes(rotation,-1,-2),inverse,volume)
                state_report['center_element_force_N']=comparison(native_force,reference_force)[0]
                old_v=np.stack([f32(n['accepted']['v_m_s']) for n in nodes]);current_v=np.stack([f32(n['center']['v_m_s']) for n in nodes])
                inertia=mass[:,None]*(current_v-old_v)
                native_force_R=dt*gather(native_force,ids,len(nodes))-inertia
                reference_force_R=dt*gather(reference_force,ids,len(nodes))-inertia
                captured_R=np.stack([f32(n['center_residual_impulse_Ns'])[:3] for n in nodes])
                state_report['center_residual_decomposition_Ns']={'native_element_gather_vs_captured_R':comparison(native_force_R[free][None],captured_R[free][None])[0],'independent_stress_gather_vs_captured_R':comparison(reference_force_R[free][None],captured_R[free][None])[0],'independent_minus_native_stress_impulse_l2':float(np.linalg.norm((reference_force_R-native_force_R)[free])),'scope':'Descriptive R=dt*sum(element forces)-m*(v-v_old), restricted to zero-gravity/no-contact coupon. Any remaining force terms or different reduction arithmetic stay visible in the native-gather comparison.'}
                arrays.update(reference_C=C,native_C=cached,reference_directional_action=analytic,reference_element_action=element,reference_mass_action=mass_action,reference_stiffness_action=stiffness_action,reference_assembled_action=total,native_assembled_action=native,free_node_mask=free)
                arrays.update(reference_center_force=reference_force,native_center_force=native_force,captured_center_R=captured_R,native_force_gather_R=native_force_R,independent_stress_gather_R=reference_force_R)
            else:
                side_index=state_index-1;epsilon=float(f32(header['epsilon_values'][side_index//2]));sign=1 if side_index%2==0 else -1
                if any(float(f32(s['epsilon']))!=epsilon or s.get('stage')!=state_index for s in states): raise ValueError('realized endpoint epsilon/stage mapping differs')
                intended=sign*epsilon*dt*direction;realized=F-centerF
                state_report['realized_endpoint_vs_ideal_F_perturbation']=comparison(realized,intended)[0]
                state_report['endpoint_relation']='captured F_side minus exact-widened center compared against sign*epsilon*dt*captured grad(delta_v); FP32 realization error is retained and is not a noise-floor qualification'
                arrays[f's{state_index}_ideal_F_delta']=intended;arrays[f's{state_index}_realized_F_delta']=realized
                signature=branch(p,old,z);center_signature=branch(p,old,center_z)
                state_report['branch_predicates_changed_from_center']=int(np.sum(np.any(signature!=center_signature,axis=1)&valid&center_valid))
                state_report['native_endpoint_relation']='Realized captured FP32 high+low endpoint, not an ideal epsilon perturbation; centered native secant remains subject to measured noise-floor and branch checks.'
            report['states'].append(state_report)
        sweep=[];qualifying=[]
        center_signature=branch(p,old,center_z)
        for power in epsilon_powers:
            epsilon=2.**(-power);ends=[]
            for sign in (1,-1):
                F=centerF[paper]+sign*epsilon*dt*direction[paper]
                z,it,rn,cn,valid,reasons,passes=project(p,F,old)
                P=np.full_like(F,np.nan);P[valid]=ref.stress(p[valid],F[valid],z[valid])
                ends.append((P,valid,branch(p,old,z)))
            plus,minus=ends;same_plus=np.all(plus[2]==center_signature,axis=1);same_minus=np.all(minus[2]==center_signature,axis=1)
            smooth=center_valid&plus[1]&minus[1]&same_plus&same_minus
            secant=(plus[0]-minus[0])/(2*epsilon*dt)
            metrics,rel,error,scale=comparison(secant,center_derivative[paper])
            good=smooth&np.isfinite(rel)&(rel<=1e-6);qualifying.append(good)
            sweep.append({'epsilon':epsilon,'power_of_two':-power,'smooth_rows':int(smooth.sum()),'rows_at_1e_6':int(good.sum()),'centered_comparison_all_rows_descriptive_only':metrics,'branch_crossing_rows':int(np.sum(center_valid&plus[1]&minus[1]&~(same_plus&same_minus))),'plus_branch_preserving_rows':int(np.sum(same_plus&center_valid&plus[1])),'minus_branch_preserving_rows':int(np.sum(same_minus&center_valid&minus[1]))})
            arrays[f'e{power}_smooth']=smooth;arrays[f'e{power}_relative_error']=rel
            # At a transition report each branch-preserving one-sided action;
            # neither side is silently promoted to a smooth centered gate.
            P0=ref.stress(p,centerF[paper],center_z)
            plus_ok=center_valid&plus[1]&same_plus;minus_ok=center_valid&minus[1]&same_minus
            sweep[-1]['one_sided_plus_branch_preserving_only']=comparison(((plus[0]-P0)/(epsilon*dt))[plus_ok],center_derivative[paper][plus_ok])[0]
            sweep[-1]['one_sided_minus_branch_preserving_only']=comparison(((P0-minus[0])/(epsilon*dt))[minus_ok],center_derivative[paper][minus_ok])[0]
            arrays[f'e{power}_plus_branch_preserving']=plus_ok;arrays[f'e{power}_minus_branch_preserving']=minus_ok
        adjacent=np.any(np.stack([a&b for a,b in zip(qualifying,qualifying[1:])]),axis=0) if len(qualifying)>1 else np.zeros(len(paper),bool)
        native_center_predicates=branch(p,old,arrays['s0_native_state'])
        inferred_agreement=np.all(native_center_predicates==center_signature,axis=1)
        report['fp64_directional_sweep']={'rows':len(paper),'rows_with_two_adjacent_declared_grid_points_at_1e_6':int(adjacent.sum()),'all_rows_have_resolved_pair':bool(adjacent.all()),'rows_with_reference_pair_and_inferred_native_center_predicate_agreement':int(np.sum(adjacent&inferred_agreement)),'native_branch_derivative_qualified':False,'grid':sweep,'scope':'Independent-reference branch only: ideal unquantized FP64 endpoints at fixed accepted history; per-tet captured direction only. Inferred native-center predicates are reported separately and are not native internal branch traces. This does not measure native FP32 noise or verify every FP64 basis column.'}
        arrays['fp64_resolved_adjacent_pair']=adjacent
        report['warnings']=[{'category':w.category.__name__,'message':str(w.message),'filename':w.filename,'line':w.lineno} for w in caught]
    np.savez_compressed(output/'arrays.npz',**arrays)
    report['limitations']=['No native finite-difference qualification without a measured absolute noise floor.','A same-center cached material/element/assembled comparison does not establish nonlinear contact or mesh/time convergence.','No damage, cohesive, rate-dependent or exact-recipe calibration is covered.']
    report['wall_seconds']=time.monotonic()-started;report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report,nonfinite=finite_report(report)
    report['nonfinite_diagnostic_fields']=nonfinite
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.bilinear-capture-layer-files.v1','files_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},'inputs_sha256':{str(p):sha(p) for p in (capture,case/'prediction.json',case/'receipt.json',case/'stderr.log',case/'output/manifest.json',Path(__file__),Path(ref.__file__),Path(geometry.__file__))}},indent=2)+'\n')
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--case',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--environment',type=int,default=0);args=ap.parse_args()
    report=audit(args.case,args.output,args.environment)
    print(json.dumps({'report':str(args.output/'report.json'),'scope':report['scope'],'fp64_sweep_all_rows':report['fp64_directional_sweep']['all_rows_have_resolved_pair'],'six_qualification_axes':report['six_qualification_axes']}))

if __name__=='__main__':main()
