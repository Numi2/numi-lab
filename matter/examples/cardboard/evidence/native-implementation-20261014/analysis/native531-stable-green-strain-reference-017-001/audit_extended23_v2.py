#!/usr/bin/env python3
import sys,json,hashlib,time,datetime,warnings
from pathlib import Path
import numpy as np
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
A=Q/'analysis/native531-strict-fp64-replay-001/layers'
NINE=Q/'analysis/native531-fp64-ninebasis-003'
PARENT=Q/'analysis/native531-all23-corrected-hill-highprecision-013-001/output-002/report.json'
CASE=Q/'solver/control17-root17-newton34-replay-531'
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
VARIANT=Q/'analysis/native531-stable-green-strain-reference-017-001/reference'
OUT=Q/'analysis/native531-stable-green-strain-reference-017-001/extended23-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(PREP));sys.path.insert(0,str(VARIANT))
import strict_layers, cardboard_bilinear_fp64_reference as ref
import cardboard_cached_operator_audit as geo
GATE=1e-6
POWERS=list(range(24,41))
def metric(a,b):
    av=np.asarray(a,dtype=np.float64).reshape(-1);bv=np.asarray(b,dtype=np.float64).reshape(-1)
    err=float(np.linalg.norm(av-bv));norm=float(np.linalg.norm(bv))
    return {'absolute_l2':err,'reference_l2':norm,'relative_l2':err/norm if norm>0 else None}
def main():
    if OUT.exists():raise SystemExit('output exists; preserve prior attempt')
    t0=time.monotonic()
    arr=np.load(A/'arrays.npz');ids=np.load(NINE/'arrays.npz')['paper_tet_ids']
    parent=json.loads(PARENT.read_text());pairs=[(int(x['tet']),int(x['basis_column'])) for x in parent['results']]
    pred=json.loads((CASE/'prediction.json').read_text());capture=Path(pred['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH'])
    header,nodes,tets,_=geo.capture_records(capture)
    tets=sorted((t for t in tets if t['environment']==0),key=lambda t:t['tet_id'])
    specs={m['index']:m for m in header['materials']};pt=strict_layers.f32(header['parameters_by_environment'][0])
    mats=np.array([t['material_index'] for t in tets]);paper=np.flatnonzero(mats!=2)
    assert len(paper)==1728 and len(ids)==1728 and len(pairs)==23
    p=np.stack([pt[specs[mats[i]]['parameter_offset']:specs[mats[i]]['parameter_offset']+28] for i in paper])
    F=arr['F_center'][paper];old=arr['accepted_paper_history'];assert np.array_equal(F,arr['F_center'][ids])
    index={int(t):i for i,t in enumerate(ids)}
    fp_powers=np.array([2.**(-k) for k in POWERS],dtype=np.float64)
    results=[]
    for tet,col in pairs:
        i=index[tet];pp=p[i:i+1];ff=F[i:i+1];oo=old[i:i+1]
        center,ci,cr,cc,cok,cwhy,cprov=strict_layers.project(pp,ff,oo)
        if not bool(cok[0]):raise RuntimeError(f'center strict projection failure tet{tet}: {cwhy[0]}')
        cb=strict_layers.branch(pp,oo,center)[0]
        h=np.zeros((1,3,3),dtype=np.float64);h.reshape(1,9)[0,col]=1.
        tan=strict_layers.strict.tangent(ref,pp,ff,oo,center,h)[0]
        pc=ref.stress(pp,ff,center)[0]
        rows=[]
        for power,eps in zip(POWERS,fp_powers):
            ends=[]
            for sign in (1.,-1.):
                fe=ff+sign*eps*h
                ze,it,rn,cn,ok,why,prov=strict_layers.project(pp,fe,oo)
                if bool(ok[0]):
                    pe=ref.stress(pp,fe,ze)[0]
                    be=strict_layers.branch(pp,oo,ze)[0]
                else:
                    pe=None;be=None
                ends.append({'valid':bool(ok[0]),'iterations':int(it[0]),'scaled_residual':float(rn[0]),'dynamic_correction':float(cn[0]),'reason':why[0],'branch':be.tolist() if be is not None else None,'state':ze[0].tolist() if bool(ok[0]) else None,'stress':pe.tolist() if pe is not None else None})
            ep,em=ends
            branch_same=bool(ep['valid'] and em['valid'] and np.array_equal(ep['branch'],cb) and np.array_equal(em['branch'],cb))
            if ep['valid'] and em['valid']:
                fd=(np.asarray(ep['stress'])-np.asarray(em['stress']))/(2.*eps)
                m=metric(fd,tan)
                plus=metric((np.asarray(ep['stress'])-pc)/eps,tan)
                minus=metric((pc-np.asarray(em['stress']))/eps,tan)
            else:m=plus=minus=None
            rows.append({'power':power,'epsilon':float(eps),'plus':ep,'minus':em,'same_branch_both_sides':branch_same,'centered_fd_vs_strict_tangent':m,'plus_one_sided_vs_tangent':plus,'minus_one_sided_vs_tangent':minus})
        good=[r for r in rows if r['same_branch_both_sides'] and r['centered_fd_vs_strict_tangent'] and r['centered_fd_vs_strict_tangent']['relative_l2'] is not None and r['centered_fd_vs_strict_tangent']['relative_l2']<=GATE]
        adj=any(a['power']+1==b['power'] for a,b in zip(good,good[1:]))
        results.append({'tet':tet,'basis_column':col,'paper_index':i,'center_iterations':int(ci[0]),'center_scaled_residual':float(cr[0]),'center_dynamic_correction':float(cc[0]),'center_branch':cb.tolist(),'tangent_norm':float(np.linalg.norm(tan)),'best_relative_error':min((r['centered_fd_vs_strict_tangent']['relative_l2'] for r in rows if r['centered_fd_vs_strict_tangent']),default=None),'best_power':min((r['power'] for r in rows if r['centered_fd_vs_strict_tangent'] and r['centered_fd_vs_strict_tangent']['relative_l2']==min(x['centered_fd_vs_strict_tangent']['relative_l2'] for x in rows if x['centered_fd_vs_strict_tangent'])),default=None),'qualifying_powers':[r['power'] for r in good],'adjacent_same_branch_pair':adj,'rows':rows})
    resolved=sum(r['adjacent_same_branch_pair'] for r in results)
    report={'schema':'numi.cardboard.native531-stable-green-strain-extended23.v2','classification':'CPU-only strict binary64 centered-FD follow-up using an isolated algebraically equivalent Green-strain evaluation; no native FP32 noise-floor or production-solver claim','bindings':{'source203_reference_sha256':sha(Q/'snapshots/source-repeated-damped-fallback-203/matter/tools/cardboard_bilinear_fp64_reference.py'),'stable_reference_sha256':sha(VARIANT/'cardboard_bilinear_fp64_reference.py'),'ninebasis_arrays_sha256':sha(NINE/'arrays.npz'),'strict_layer_arrays_sha256':sha(A/'arrays.npz'),'strict_layer_manifest_sha256':sha(A/'manifest.json'),'corrected_mp_013_report_sha256':sha(PARENT),'capture_sha256':sha(capture),'prediction_sha256':sha(CASE/'prediction.json'),'reader_sha256':sha(PREP/'strict_layers.py')},'scope':{'pairs':pairs,'pair_count':len(pairs),'epsilon_powers':POWERS,'perturbation':'exact binary64 F +/- 2^-power * unit local-F basis direction; strict endpoints reprojected from exact accepted history','residual_gate':2e-12,'dynamic_correction_gate':2e-11,'FD_relative_gate':GATE,'material_rate_model':'same captured rate-independent paper law; no Fdot enters constitutive equations'},'summary':{'resolved_adjacent_same_branch_pairs':resolved,'total_pairs':len(results),'unresolved_pairs':len(results)-resolved,'unresolved':[{k:r[k] for k in ('tet','basis_column','best_relative_error','best_power','qualifying_powers')} for r in results if not r['adjacent_same_branch_pair']]},'results':results,'limits':['No tolerance changed. A pair resolves only with adjacent powers, both strict projections passing both unchanged gates, and both endpoints retaining the center branch.','This is same-source FP64 reference self-consistency only. It does not establish a native FP32 absolute noise floor or production constitutive qualification.','The stable strain form is algebraically identical to F.T F-I but changes rounding. The original source and coarse 10..24 sweep remain retained separately.','Every selected pair and every endpoint is retained; no failed row is omitted.'],'wall_seconds':time.monotonic()-t0,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (OUT/'run.json').write_text(json.dumps({'argv':[str(Q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(__file__)],'cwd':str(Q),'cpu_only':True,'powers':POWERS,'gate':GATE},indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'resolved':resolved,'total':len(results),'unresolved':report['summary']['unresolved'],'wall_seconds':report['wall_seconds']},indent=2))
if __name__=='__main__':main()
