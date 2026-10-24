"""Pinned all-face regression for retained knee/ankle/MTP poses; source prep only."""
import sys, json, hashlib, importlib.util, time
from pathlib import Path
import numpy as np

R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
OUT=E/'remaining-joint-global-regression-251'
POSE=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001/joint-surface-audit-001/poses')
NAMES=['knee_bilateral_q106q120_0p25','ankle_bilateral_positive_q109q123_0p25','ankle_bilateral_negative_q109q123_m0p25','mtp_bilateral_positive_q111q125_0p25','mtp_bilateral_negative_q111q125_m0p25']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('fixedroi251','/tmp/fixed_rest_seam_roi_002.py')
f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f)
from numilab_human.cardiac_cavity_intersections import _audit_self_pairs_touching_faces

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir()
    p=f.RB.decode(f.BASE.read_bytes()); F=np.asarray(p['inds'],dtype=np.int64)
    assert sha(f.BASE)==f.m.EXPECTED_BASE
    neutral=f.REST/'registered-skin-poses.json'
    A,b=f.RB.reference_transforms(p,f.m.pose(json.loads(neutral.read_text()),'current'),[0,0,0])
    Xref=f.RB.positions(p,A,b).astype('<f4')
    qmap,members,classes,valid,*_=f.source_map(F,Xref)
    metadata={'source':str(f.BASE),'source_sha256':sha(f.BASE),'face_count':len(F),
      'neutral_pose_sha256':sha(neutral),'source_seam_rule':'fixed valid source-rest classes only; no posed-coordinate welding',
      'helper_sha256':sha('/tmp/fixed_rest_seam_roi_002.py'), 'rebind_sha256':sha(f.m.RBPATH),
      'script_sha256':sha(__file__),'results':[],
      'scope':'Exact all-face self-crossing regression at retained discrete kinematic poses. No mechanics, contact, continuous ROM, or physical admission.'}
    for name in NAMES:
        path=POSE/(name+'.json'); snap=json.loads(path.read_text())
        A,b=f.RB.reference_transforms(p,snap,[0,0,0]);X=f.RB.positions(p,A,b).astype('<f4')
        if not np.isfinite(X).all(): raise ValueError('nonfinite emitted pose')
        eff,em=f.effective_source_ids(qmap,members,valid,X)
        records,deg=f.exact_records(X,F,eff)
        started=time.monotonic(); aud=_audit_self_pairs_touching_faces(records,set(range(len(F))))
        pairs=sorted(tuple(map(int,q)) for q in aud['triangle_pairs'])
        row={'name':name,'pose_path':str(path),'pose_sha256':sha(path),
           'world_float32_sha256':hashlib.sha256(X.tobytes()).hexdigest(),
           'selected_face_count':len(F),'complete_surface_face_count':len(F),
           'complete_surface_audit':True,'nondegenerate_faces':len(records),
           'degenerate_faces_excluded':int(deg),'seams':em,'triangle_pairs':[list(q) for q in pairs],
           'crossings':len(pairs),'aabb_candidates':int(aud['aabb_candidate_pairs']),
           'seconds':time.monotonic()-started}
        metadata['results'].append(row)
        (OUT/'partial.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(name,'global_crossings',len(pairs),'degenerate',deg,'seconds',round(row['seconds'],3),flush=True)
    (OUT/'report.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('report_sha256',sha(OUT/'report.json'),flush=True)

if __name__=='__main__':main()
