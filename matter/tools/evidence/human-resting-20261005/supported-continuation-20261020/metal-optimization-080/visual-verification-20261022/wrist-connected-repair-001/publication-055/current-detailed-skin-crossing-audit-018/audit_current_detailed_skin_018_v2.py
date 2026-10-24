"""Exact current detailed-skin/implicated-bank audit using existing owner315."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
sys.path.insert(0,str(E/'global-inner-p1-full-domain-clipping-009/exact-O-pair-011'))
from audit_exact_pairs_011 import ci,classify,records
sys.path.insert(0,str(E/'registered-pose-frame-helper-194/tools'))
import rebind_human_skin_reference as rb
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_native_positions(path,n):
    d=np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding=None)
    if len(d)!=n or not np.array_equal(d['vertex'],np.arange(n)):raise ValueError('native vertex order mismatch')
    x=np.column_stack([d[k] for k in ('wx','wy','wz')])
    if not np.array_equal(x,x.astype(np.float32).astype(float)):raise ValueError('positions not exact Float32')
    return x.astype(np.float32)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(exist_ok=False)
    begun=time.monotonic()
    skin=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin')
    pose=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json'
    launch=pose.with_name('launch-manifest.snapshot.json')
    csv=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin.csv'
    zp=E/'global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz'
    bank=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz';ip=bank.with_name('inventory.json')
    assert sha(skin)=='d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04'
    assert sha(csv)=='eb4cbae46b88d82e1f40ab6429cb273263f781811ce52ab3d4f2543ce435270e'
    payload=rb.decode(skin.read_bytes());f=np.asarray(payload['inds'],np.int64)
    x=rb.registered_positions_float32(payload,json.loads(pose.read_text()),frame='current')
    previous_native=load_native_positions(csv,len(payload['verts']))
    discrepancy=np.linalg.norm(x.astype(float)-previous_native.astype(float),axis=1)
    prior_manifest=json.loads(launch.read_text());argv=prior_manifest['argv']
    prior_skin=Path(argv[argv.index('--skin-payload')+1])
    prior_csv=np.genfromtxt(csv,delimiter=',',names=True,dtype=None,encoding=None)
    current_donor=payload['binds']['body'][payload['weights'].argmax(1)]
    sr=records(x,f,np.arange(len(f)));skinindex=ci._prepare_surface_aabb(tuple(sr.values()))
    z=dict(np.load(zp));banks=np.load(bank);inv=json.loads(ip.read_text())['banks']
    hits=[];offset=[0];world=[];codes=[];names=[];per=[];allclasses=Counter();total_candidates=0
    for bi in np.unique(z['bank_ids']):
        bi=int(bi);meta=inv[bi];key=meta['key']
        ids=np.unique(z['bank_local_face_ids'][z['bank_ids']==bi])
        v=banks[key+'_vertices'];bf=banks[key+'_faces']
        br=records(v,bf,ids)
        result=ci._audit_pair_prepared_first(skinindex,tuple(br.values()),same_surface=False)
        counts=Counter();total_candidates+=result['aabb_candidate_pairs']
        for si,bfi in result['triangle_pairs']:
            name,points=classify(sr[si][0],br[bfi][0]);assert name!='clear'
            counts[name]+=1;allclasses[name]+=1
            if name not in names:names.append(name)
            hits.append((bi,int(bfi),int(si)));codes.append(names.index(name))
            world.extend(tuple(float(v/(1<<149)) for v in p) for p in points);offset.append(len(world))
        per.append(dict(bank_id=bi,key=key,kind=meta['kind'],stable_id=meta['stable_id'],layer=meta['layer'],
            body_indices=meta['body_indices'],selected_bank_facets=len(ids),aabb_candidates=result['aabb_candidate_pairs'],
            exact_intersecting_pairs=result['count'],classes=dict(counts)))
        print(json.dumps(per[-1]),flush=True)
    rows=np.asarray(hits,np.int64).reshape(-1,3);dest=out/'detailed-skin-exact-pairs.npz'
    np.savez_compressed(dest,bank_ids=rows[:,0],bank_local_face_ids=rows[:,1],detailed294_face_ids=rows[:,2],
        class_codes=np.asarray(codes,np.uint8),class_names=np.asarray(names),exact_hit_offsets=np.asarray(offset,np.int64),
        exact_hit_points_m=np.asarray(world).reshape(-1,3))
    report=dict(status='EXACT_OWNER194_FLOAT32_REPLAYED_294_VS_ORIGINAL_IMPLICATED_BANK_FACETS',detailed_skin_vertices=len(x),detailed_skin_faces=len(f),
        selected_banks=len(per),selected_bank_facets=sum(i['selected_bank_facets'] for i in per),
        complete_detailed_skin_side=True,complete_tissue_bank_side=False,aabb_candidates=total_candidates,
        exact_intersecting_pairs=len(rows),classes=dict(allclasses),per_bank=per,
        interpretation='Tests owner194 Float32 source294 replay triangles at the explicitly pinned recorded pose. This is a preparation replay, not a native294 capture; the previous native CSV used a different skin payload. A transverse intersection proves a source-surface conflict in this pose. Absence from this facet subset does not certify whole-bank clearance.',
        anatomical_interface_exemptions_applied=False,native_capture_qualification=False,
        geometry_source='owner194 registered_positions_float32 on pinned294 at identity004 body poses; no native CSV positions or donor labels used as replay geometry',
        geometry_array_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        previous_native_comparison=dict(declared_payload=str(prior_skin),payload_sha256=sha(prior_skin),
            capture_payload_matches_294=sha(prior_skin)==sha(skin),
            bytewise_changed_vertices=int(np.any(x.view(np.uint32)!=previous_native.view(np.uint32),axis=1).sum()),
            maximum_position_difference_m=float(discrepancy.max()),
            donor_label_disagreement_count=int((prior_csv['donor']!=current_donor).sum()),
            csv_matrix_columns='world_to_donor_local_linear, not the skinning Jacobian'),
        input_sha256={str(p):sha(p) for p in (skin,csv,pose,launch,prior_skin,zp,bank,ip,Path(__file__),Path(ci.__file__),Path(rb.__file__))},
        output_sha256=sha(dest),elapsed_s=time.monotonic()-begun,physical_admission=False,native_admission=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_bank'}),flush=True)
if __name__=='__main__':main()
