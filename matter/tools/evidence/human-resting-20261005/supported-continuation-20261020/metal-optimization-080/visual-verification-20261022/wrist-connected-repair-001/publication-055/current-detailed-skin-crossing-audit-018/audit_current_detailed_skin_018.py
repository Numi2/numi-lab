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
    csv=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin.csv'
    zp=E/'global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz'
    bank=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz';ip=bank.with_name('inventory.json')
    assert sha(skin)=='d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04'
    assert sha(csv)=='eb4cbae46b88d82e1f40ab6429cb273263f781811ce52ab3d4f2543ce435270e'
    payload=rb.decode(skin.read_bytes());f=np.asarray(payload['inds'],np.int64)
    x=load_native_positions(csv,len(payload['verts']))
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
    report=dict(status='EXACT_CURRENT_DETAILED_SKIN_VS_ORIGINAL_IMPLICATED_BANK_FACETS',detailed_skin_vertices=len(x),detailed_skin_faces=len(f),
        selected_banks=len(per),selected_bank_facets=sum(i['selected_bank_facets'] for i in per),
        complete_detailed_skin_side=True,complete_tissue_bank_side=False,aabb_candidates=total_candidates,
        exact_intersecting_pairs=len(rows),classes=dict(allclasses),per_bank=per,
        interpretation='Tests actual native uploaded source294 triangles, not the mechanical O chords or nearest points. A transverse intersection proves a source-surface conflict in this pose. Absence from this facet subset does not certify whole-bank clearance.',
        anatomical_interface_exemptions_applied=False,input_sha256={str(p):sha(p) for p in (skin,csv,zp,bank,ip,Path(__file__),Path(ci.__file__),Path(rb.__file__))},
        output_sha256=sha(dest),elapsed_s=time.monotonic()-begun,physical_admission=False,native_admission=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_bank'}),flush=True)
if __name__=='__main__':main()
