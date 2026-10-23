from pathlib import Path
import sys, json, hashlib, shutil
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import wrist_sections_20261022 as ws
E=ws.E
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.skin_source_payload_preflight import decode_payload
BASE=E/'finger-profile-068-whole-hand-minimax'
LEFT=E/'finger-profile-069d-full-ancestry-clearance'
STATE=E/'finger-profile-069g-replay-state'
OUTNAME=sys.argv[1] if len(sys.argv)>1 else 'finger-profile-069h-reproduced-001'
if Path(OUTNAME).name!=OUTNAME: raise ValueError('pass only a fresh output folder name')
OUT=E/OUTNAME
OUT.mkdir(exist_ok=False)
pose=json.loads(next((E/'native-pose-checks/repair-005/neutral').glob('*.skin-poses.json')).read_text())
raw_left=(LEFT/'replacement-bilateral-hands038.nhskin').read_bytes()
Dleft=decode_payload(raw_left)
raw_base=(BASE/'replacement-bilateral-hands038.nhskin').read_bytes()
Dbase=decode_payload(raw_base)
st=np.load(STATE/'right-scale020-proposal-state.npz')
slots=np.asarray(st['source_slots'],int); delta=np.asarray(st['delta_world'],float); proposal=np.asarray(st['proposal_world'],float); baseworld=np.asarray(st['base_world'],float); faces_patch=np.asarray(st['faces'],int)
# State must be an exact replay from the original 068 right-side patch.
basepatch=np.load(BASE/'right-patch.npz')
assert np.array_equal(slots,np.asarray(basepatch['source_slots'],int))
assert np.array_equal(baseworld,np.asarray(basepatch['world'],float))
assert np.allclose(proposal,baseworld+delta,rtol=0,atol=1e-14)
# 069d changes the left side only; right side remains the exact 068 parent at the shared source slots.
Qbase=ws.ns['posed'](Dbase,pose).astype(np.float64)
Qleft=ws.ns['posed'](Dleft,pose).astype(np.float64)
right_parent_error=float(np.max(np.linalg.norm(Qleft[slots]-baseworld,axis=1)))
if right_parent_error>2e-6: raise RuntimeError(('069d right does not match 068 right baseline',right_parent_error))
Q=Qleft.copy(); moved=np.flatnonzero(np.linalg.norm(delta,axis=1)>1e-12); changed=np.asarray(sorted(set(map(int,slots[moved]))),int); Q[slots]+=delta
# The scaled patch is subject to the recorded physical geometry gates; only the source-angle preference is waived.
fit=json.loads((STATE/'build-report.json').read_text())
scale_attempt=next(a for a in fit['sides']['right']['attempts'] if abs(a['criteria']['scale']-.2)<1e-10)
c=scale_attempt['criteria']
assert c['face_orientation_area_pass'] and c['bone_barrier_pass'] and c['move_cap_pass'] and c['boundary_move_mm']==0.0
assert c['new_severe_edges_over90']==0 and c['severe_edges_before_after'][1] <= c['severe_edges_before_after'][0]
assert not c['target_source_fit_pass']
# Build current-pose inverse LBS from the exact parent NHSKIN, then rewrite only changed right-hand positions.
W=np.asarray(Dleft['full_weights'],float); BF=np.asarray(Dleft['bindings_f'],float); bindings=np.asarray(Dleft['bindings_u'][:,0],int)
state={int(x['body_index']):x for x in pose['bodies']}; nvert=len(Q)
A=np.zeros((nvert,3,3),float); N=np.zeros_like(A); tr=np.zeros_like(Q)
for j,b in enumerate(bindings):
 stbody=state[int(b)];rc=ws.ns['mat'](stbody['current']['orientation_world_xyzw']);restR=ws.ns['mat'](stbody['rest']['orientation_world_xyzw'])
 A+=W[:,j,None,None]*(rc*BF[j,8]@ws.ns['mat'](BF[j,4:8]))[None]
 tr+=W[:,j,None]*(BF[j,1:4]@rc.T+stbody['current']['position_world_m'])
 N+=W[:,j,None,None]*(rc@restR.T)[None]
F=np.asarray(Dleft['indices'],int).reshape(-1,3)
raw=bytearray(raw_left); arr=np.ndarray((nvert,14),dtype='<f4',buffer=raw,offset=60+36*Dleft['binding_count'])
if len(changed): arr[changed,:3]=np.linalg.solve(A[changed],(Q[changed]-tr[changed])[:,:,None])[:,:,0]
affected=np.unique(F[np.any(np.isin(F,changed),axis=1)]) if len(changed) else np.array([],int)
fn=np.cross(Q[F[:,1]]-Q[F[:,0]],Q[F[:,2]]-Q[F[:,0]])
wn=np.zeros_like(Q);np.add.at(wn,F.ravel(),np.repeat(fn,3,axis=0));wn/=np.maximum(np.linalg.norm(wn,axis=1)[:,None],1e-30)
if len(affected):
 ln=np.linalg.solve(N[affected],wn[affected,:,None])[:,:,0];ln/=np.maximum(np.linalg.norm(ln,axis=1)[:,None],1e-30);arr[affected,3:6]=ln
payload=bytes(raw); (OUT/'replacement-bilateral-hands038.nhskin').write_bytes(payload)
D=decode_payload(payload);Qcheck=ws.ns['posed'](D,pose).astype(np.float64)
reproject=float(np.max(np.linalg.norm(Qcheck[changed]-Q[changed],axis=1))) if len(changed) else 0.
assert reproject<2e-6
meta_exact=all(np.array_equal(D[k],Dleft[k]) for k in ('indices','bindings_f','bindings_u','full_weights'))
assert meta_exact
# Preserve all left-side positions exactly, and keep right boundary/seam positions exactly at 068/069d.
left_patch=np.load(LEFT/'left-patch.npz'); left_slots=np.asarray(left_patch['source_slots'],int)
left_pos_bit_exact=bool(np.array_equal(D['vertices_f'][left_slots,:3],Dleft['vertices_f'][left_slots,:3]))
assert left_pos_bit_exact
edges={}
for fi,tri in enumerate(faces_patch):
 for a,b in ((int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))):edges.setdefault(tuple(sorted((a,b))),[]).append(fi)
boundary=np.asarray(sorted({v for e,fs in edges.items() if len(fs)==1 for v in e}),int)
boundary_move_mm=float(np.max(np.linalg.norm(proposal[boundary]-baseworld[boundary],axis=1))*1000) if len(boundary) else 0.
assert boundary_move_mm<1e-10
# Store the right patch in the original 068 patch schema and the exact fit state separately.
patch_fields={k:basepatch[k].copy() for k in basepatch.files};patch_fields['world']=proposal
np.savez_compressed(OUT/'right-patch.npz',**patch_fields)
shutil.copy2(LEFT/'left-patch.npz',OUT/'left-patch.npz')
shutil.copy2(STATE/'right-scale020-proposal-state.npz',OUT/'right-scale020-proposal-state.npz')
# Keep the exact geometry vector file format used by the parent audit.
g0=np.load(LEFT/'replacement-geometry.npz'); gg={k:g0[k] for k in g0.files};gg['world']=Qcheck;np.savez(OUT/'replacement-geometry.npz',**gg)
# Confirm the final right candidate inherits 069d everywhere outside the 1603 moved surface rows.
unchanged=np.setdiff1d(np.arange(nvert),changed)
unchanged_positions_bit_exact=bool(np.array_equal(D['vertices_f'][unchanged,:3],Dleft['vertices_f'][unchanged,:3]))
assert unchanged_positions_bit_exact
report={
 'schema':'numi.human.finger-diagnostic-candidate.v1',
 'candidate':'069h-right-scale020-preference-waiver',
 'status':'diagnostic_candidate_emitted_pending_full_audit',
 'admission':'not_physically_qualified; exact full-surface, collision, functional-pose, and mechanical-layer audits remain required',
 'parent_069d_payload_sha256':hashlib.sha256(raw_left).hexdigest(),
 'parent_068_payload_sha256':hashlib.sha256(raw_base).hexdigest(),
 'candidate_payload_sha256':hashlib.sha256(payload).hexdigest(),
 'global_index_sha256':hashlib.sha256(np.asarray(D['indices'],dtype='<u4').tobytes()).hexdigest(),
 'fit_script':'build-069g-replay-save-state.py',
 'fit_status':{'unconstrained_lsq_success':bool(np.load(STATE/'right-unconstrained-lsq-state.npz')['success']),'slsqp_success':bool(st['solver_success']),'slsqp_message':str(st['solver_message'].item()),'slsqp_nfev':int(st['solver_nfev']),'slsqp_cost':float(st['solver_cost'])},
 'right_scale':0.2,
 'source_angle_preference':{'status':'explicitly_waived_for_this_diagnostic_only','reason':'scale 0.2 lowers the right crease from 91.1139 to 89.5863 degrees but increases absolute deviation from registered source angle 118.7565 degrees from 27.6426 to 29.1703 degrees (+1.52765 degrees); this is a source-fit construction preference, not an anatomical or physical acceptance gate.'},
 'geometry_gates_from_exact_069g_scale020':{k:c[k] for k in ('new_severe_edges_over90','severe_edges_before_after','max_severe_angle_before_after_deg','target_angles_before_after_deg','target_source_error_before_after_deg','min_changed_face_area_ratio','min_changed_face_normal_dot','boundary_move_mm','min_control_clearance_after_mm','clearance_violating_controls','max_vertex_move_mm','face_orientation_area_pass','bone_barrier_pass','move_cap_pass')},
 'assembly_integrity':{'left_positions_bit_exact_to_069d':left_pos_bit_exact,'all_unchanged_global_positions_bit_exact_to_069d':unchanged_positions_bit_exact,'right_parent_error_to_068_m':right_parent_error,'right_boundary_move_mm':boundary_move_mm,'pose_reprojection_max_error_m':reproject,'indices_bindings_weights_bit_exact_to_069d':meta_exact,'changed_global_rows':int(len(changed)),'affected_normal_rows':int(len(affected))},
 'wrist_scope':'The candidate starts from the complete 069d payload, retains its left wrist byte-for-byte, and applies only the right 069g scale-0.2 patch with its boundary pinned. Full wrist/pose/intersection audit remains pending.',
 'not_claimed':['physical qualification','all-pose collision clearance','full-surface self-contact admission','mechanical layer feasibility']
}
(OUT/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'candidate_sha256':report['candidate_payload_sha256'],'changed_global_rows':len(changed),'affected_normal_rows':len(affected),'left_exact':left_pos_bit_exact,'unchanged_exact':unchanged_positions_bit_exact,'pose_reprojection_max_error_m':reproject,'meta_exact':meta_exact,'right_parent_error_m':right_parent_error,'scale020':c['scale']},indent=2))
