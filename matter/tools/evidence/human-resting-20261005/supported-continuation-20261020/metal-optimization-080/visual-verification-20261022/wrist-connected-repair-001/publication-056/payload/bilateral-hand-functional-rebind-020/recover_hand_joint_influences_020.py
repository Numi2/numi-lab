"""Recover historical hand joint influences with current294 rest shape preserved.

Offline source compilation only, through existing owner194. No forces, locks,
accepted mechanics, or native qualification are introduced here.
"""
from pathlib import Path
import hashlib,json,sys,argparse
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
sys.path.insert(0,str(E/'registered-pose-frame-helper-194/tools'))
import rebind_human_skin_reference as rb
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def recover_hand_rows(current,historical):
    if current['nv']!=historical['nv'] or not np.array_equal(current['binds'],historical['binds']):
        raise ValueError('source vertex domain or binding identities differ')
    if current['source_sha256']!=historical['source_sha256'] or current['fingerprint']!=historical['fingerprint']:
        raise ValueError('registered source lineage differs')
    bodies=current['binds']['body'].astype(int)
    donor=bodies[current['weights'].argmax(1)]
    result=current['weights'].copy();selected=np.zeros(current['nv'],bool);side_rows={}
    for wrist,finger_lo,finger_hi,label in ((49,50,76,'right'),(99,100,126,'left')):
        hand=(bodies==wrist)|((bodies>=finger_lo)&(bodies<=finger_hi))
        hand_support=historical['weights'][:,hand].astype(float).sum(1)
        finger_support=historical['weights'][:,(bodies>=finger_lo)&(bodies<=finger_hi)].astype(float).sum(1)
        # Source-supported hand rows only; do not change neighboring forearm,
        # elbow, shoulder, toe, or trunk owners as a side effect.
        rows=(donor==wrist)&(hand_support>=.95)&(finger_support>1e-5)
        result[rows]=historical['weights'][rows];selected|=rows
        side_rows[label]=np.flatnonzero(rows).tolist()
    rb.validate_weights(result)
    return result,selected,side_rows
def reference_palette(payload,snapshot,frame):
    rows={r['body_index']:r for r in snapshot['all_bodies']}
    source=[]
    for b in payload['binds']:
        body=int(b['body']);r=rows[body][frame]
        source.append(dict(core_body=body,
            quaternion_xyzw=np.asarray(r['orientation_world_xyzw'],np.float32).astype(float).tolist(),
            position=np.asarray(r['position_world_m'],np.float32).astype(float).tolist()))
    return rb.reference_transforms(payload,dict(source_poses=source),[0.,0.,0.])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(exist_ok=False)
    current_path=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin')
    history_path=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin'
    pose_path=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json'
    if sha(current_path)!='d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04':raise ValueError('294 hash changed')
    if sha(history_path)!='4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5':raise ValueError('128 hash changed')
    current=rb.decode(current_path.read_bytes());historical=rb.decode(history_path.read_bytes());snapshot=json.loads(pose_path.read_text())
    weights,selected,sides=recover_hand_rows(current,historical)
    A,B=reference_palette(current,snapshot,'current');blob,diag=rb.rebind(current,weights,A,B)
    candidate=rb.decode(blob)
    before=rb.registered_positions_float32(current,snapshot,frame='current')
    after=rb.registered_positions_float32(candidate,snapshot,frame='current')
    error=np.linalg.norm(after.astype(float)-before.astype(float),axis=1)
    if error.max()>2e-6:raise ValueError('emitted reference geometry exceeds 2um gate')
    if not np.array_equal(candidate['inds'],current['inds']):raise ValueError('topology changed')
    if not np.array_equal(candidate['verts'][~selected],current['verts'][~selected]):raise ValueError('outside source vertex bytes changed')
    if not np.array_equal(candidate['weights'][~selected],current['weights'][~selected]):raise ValueError('outside weights changed')
    path=args.out/'candidate.nhskin';path.write_bytes(blob)
    np.savez_compressed(args.out/'selected-hand-rows.npz',source_vertex_ids=np.flatnonzero(selected),
        right_source_vertex_ids=np.asarray(sides['right']),left_source_vertex_ids=np.asarray(sides['left']),
        current_reference_positions_world_m=before,rest_preserved_positions_world_m=after)
    bodies=current['binds']['body']
    coverage={}
    for label,ids in sides.items():
        ids=np.asarray(ids);old=current['weights'][ids];new=candidate['weights'][ids]
        coverage[label]=dict(rows=len(ids),before_active_body_ids=bodies[np.any(old>1e-5,axis=0)].tolist(),
            after_active_body_ids=bodies[np.any(new>1e-5,axis=0)].tolist(),
            before_dominant_body_ids=np.unique(bodies[old.argmax(1)]).tolist(),
            after_dominant_body_ids=np.unique(bodies[new.argmax(1)]).tolist())
    report=dict(status='UNADMITTED_REST_PRESERVING_HAND_INFLUENCE_RECOVERY',source_vertex_count=current['nv'],
        changed_weight_rows=int(selected.sum()),coverage=coverage,owner_rebind=diag,
        emitted_reference_error_max_m=float(error.max()),emitted_reference_error_rms_m=float(np.sqrt(np.mean(error**2))),
        unchanged_outside_source_vertices=True,unchanged_outside_weights=True,unchanged_face_indices=True,
        target_pose='identity004 current body poses applied to current294; this is source replay not native294 qualification',
        provenance='Historical128 inferred binding weights on existing source vertex IDs, selectively restored; current294 topology/registered reference contour retained. Not new measured anatomy.',
        limitations=['No new physical constraints or wrist locks.','Historical weights are a candidate, not independent anatomy validation.',
            'Normals retained from294; native normal/pose confirmation remains required.',
            'Requires source transfer/mass/contact rebuild and multi-pose exact crossing checks before use in native scene.'],
        input_sha256={str(p):sha(p) for p in (current_path,history_path,pose_path,Path(rb.__file__),Path(__file__))},
        candidate_sha256=sha(path),physical_admission=False,native_admission=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('owner_rebind','input_sha256')},indent=2))
if __name__=='__main__':main()
