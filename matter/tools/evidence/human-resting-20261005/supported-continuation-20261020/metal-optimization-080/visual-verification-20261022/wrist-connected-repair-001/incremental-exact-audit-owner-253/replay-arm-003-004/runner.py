import sys, json, time, hashlib, importlib.util
from pathlib import Path
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
O=E/'incremental-exact-audit-owner-253/replay-arm-003-004'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False)
    f=load('fixedroi253','/tmp/fixed_rest_seam_roi_002.py')
    ci=load('numilab_human.cardiac_cavity_incremental253',E/'incremental-exact-audit-owner-253/src/numilab_human/cardiac_cavity_intersections.py')
    J=E/'joint-functional-owner-audit-20261023'
    a=J/'original128-local-vertex-fit-003-conservative-area/candidate.nhskin'
    b=J/'original128-local-vertex-fit-004-refreshed/candidate.nhskin'
    assert sha(a)=='b0e2b052e2e4552767c500a93e99ca03f75fcdedccdbd4dbe8b225b5dd07ae8c'
    assert sha(b)=='0de586c667f5ba6155364d95efa3ab20250c31d41ef807f62d6d8d868cb2c305'
    base=f.RB.decode(f.BASE.read_bytes());F=np.asarray(base['inds'],dtype=np.int64)
    neutral=f.REST/'registered-skin-poses.json'
    A0,b0=f.RB.reference_transforms(base,f.m.pose(json.loads(neutral.read_text()),'current'),[0,0,0])
    ref=f.RB.positions(base,A0,b0).astype('<f4')
    qmap,members,classes,valid,*_=f.source_map(F,ref)
    pose=f.m.POSES['shoulder_elevation']/'registered-skin-poses.json'
    snap=json.loads(pose.read_text());rows=[];emissions=[]
    for p in (a,b):
        data=f.RB.decode(p.read_bytes());assert data['inds'].tobytes()==base['inds'].tobytes()
        A,B=f.RB.reference_transforms(data,f.m.pose(snap,'current'),[0,0,0]);X=f.RB.positions(data,A,B).astype('<f4')
        eff,metadata=f.effective_source_ids(qmap,members,valid,X)
        records,deg=f.exact_records(X,F,eff);assert deg==0
        rows.append(records);emissions.append(hashlib.sha256(X.tobytes()).hexdigest())
    start=time.monotonic();cache,baseline=ci._prepare_complete_self_pair_baseline(rows[0]);base_seconds=time.monotonic()-start
    start=time.monotonic();cache,incremental=ci._update_complete_self_pair_baseline(cache,rows[1]);incremental_seconds=time.monotonic()-start
    start=time.monotonic();full=ci._audit_self_pairs_touching_faces(rows[1],set(range(len(F))));full_seconds=time.monotonic()-start
    assert incremental['triangle_pairs']==full['triangle_pairs']
    assert baseline['count']==25 and full['count']==30
    report={'schema':'numi.human.incremental-exact-self-audit-replay.v1','inputs':{'candidate003':sha(a),'candidate004':sha(b),'pose':sha(pose),'rebind_helper':sha(f.m.RBPATH),'exact_predicates':sha(ci.__file__),'script':sha(__file__)},'world_float32_sha256':emissions,'face_count':len(F),'baseline_count':baseline['count'],'candidate_count':full['count'],'pair_sets_exactly_equal':True,'baseline_seconds':base_seconds,'full_candidate_seconds':full_seconds,'incremental_candidate_seconds':incremental_seconds,'incremental_report':incremental,'full_candidate_report':full,'scope':'Static source-preparation audit only. Both candidates fail anatomy. No physical simulation throughput claim.'}
    (O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('incremental_report','full_candidate_report')},indent=2),flush=True)
if __name__=='__main__':main()
