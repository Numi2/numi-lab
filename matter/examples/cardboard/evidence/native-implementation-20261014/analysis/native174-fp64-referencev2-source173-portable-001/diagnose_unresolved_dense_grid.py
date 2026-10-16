#!/usr/bin/env python3
"""Focused quarter-power FP64 FD follow-up for run018's 16 unresolved rows."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

SOURCE_SHA = "a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995"
CAPTURE_SHA = "4361b6aa9f0d269a9733b0bd8f508dbc1204624451611816512dad02c303d27c"
MP_SHA = "107e942cc2e131eed8f8db7759f8deb3e86fcdc42d8da249996daf51d6ed11ad"
ARR018_SHA = "626ad3fe6d6bd147b3b18e8d8a688d63a430a43af33c2aeb9ca4f103a8811808"
STRICT_R = 2.0e-12
STRICT_C = 2.0e-11
DERIVATIVE_GATE = 1.0e-6
# The coarse run's 16 unresolved cases mostly bottomed out at its largest
# exponent (26).  This symmetric dense follow-up brackets those apparent
# minima on both sides and extends toward smaller representable perturbations.
GRID = tuple(20.0 + 0.25*k for k in range(49))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    return mod

def project(base,ref,p,F,old):
    return base.continue_projection_tight(ref,p,F,old,p[:,26],
        base.project_centers(ref,p,F,old,p[:,26]))

def signature(v2,p,old,z):
    _,parts,meta=v2.fb_row(p,old,z)
    return np.stack((np.sign(parts[2].rounded()),np.sign(parts[3].rounded()),
                     meta['fb_quotient_branch'].astype(np.float64)),axis=1)

def metric(obs,ref):
    diff=float(np.linalg.norm(obs-ref)); norm=float(np.linalg.norm(ref))
    return (diff/norm if norm>0 else None,diff,norm)

def scalar(x):
    x=float(x); return x if np.isfinite(x) else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-reference',required=True,type=Path)
    ap.add_argument('--capture',required=True,type=Path)
    ap.add_argument('--mp-arrays',required=True,type=Path)
    ap.add_argument('--portable-tools',required=True,type=Path)
    ap.add_argument('--run018',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.output.exists(): raise SystemExit(f'refusing overwrite: {a.output}')
    expected=((a.source_reference,SOURCE_SHA),(a.capture,CAPTURE_SHA),(a.mp_arrays,MP_SHA),
              (a.run018/'nine-basis-fd-arrays.npz',ARR018_SHA))
    for p,want in expected:
        got=sha(p)
        if got!=want: raise SystemExit(f'hash mismatch {p}: {got} != {want}')
    tools=a.portable_tools.resolve(); sys.path.insert(0,str(tools))
    base_loader=load('cardboard_compensated_fp64_base',tools/'cardboard_compensated_fp64_base.py')
    load('cardboard_dd_fp64',tools/'cardboard_dd_fp64.py')
    v2=load('cardboard_compensated_fp64_reference',tools/'cardboard_compensated_fp64_reference.py')
    base=base_loader.import_reference(a.source_reference); ref=v2.Reference(base)
    cap=np.load(a.capture); mpa=np.load(a.mp_arrays); coarse=np.load(a.run018/'nine-basis-fd-arrays.npz')
    summary=json.loads((a.run018/'report.json').read_text())
    miss=[]
    for d in range(9):
        for row in np.flatnonzero(~coarse['adjacent_pair_pass'][d]):
            miss.append((d,int(row),int(coarse['paper_tets'][row])))
    if len(miss)!=16: raise SystemExit(f'run018 unresolved set changed: {len(miss)}')
    paper_tets=np.flatnonzero(cap['material']!=2)
    if len(paper_tets)!=1728 or not np.array_equal(mpa['paper_material_index'],cap['material'][paper_tets]):
        raise SystemExit('paper rows/material ordering changed')
    p=mpa['paper_parameters_f32'].astype(np.float64)
    old=cap['accepted_paper_history'].astype(np.float64)
    F0=cap['F_center'][paper_tets].astype(np.float64)
    zall=np.load(a.run018/'center-projection.npz')['state']
    center_out=np.load(a.run018/'center-implicit-tangent.npz')
    if not np.array_equal(center_out['paper_tets'],paper_tets): raise SystemExit('run018 tangent row ordering mismatch')
    C=center_out['tangent']; z0=zall
    s0=base.stress(p,F0,z0).reshape(1728,9)
    sig0=signature(v2,p,old,z0)
    rows=np.asarray([row for _,row,_ in miss],dtype=np.int64)
    dirs=np.asarray([d for d,_,_ in miss],dtype=np.int64)
    tets=np.asarray([tet for _,_,tet in miss],dtype=np.int64)
    pp=p[rows]; oo=old[rows]; ff=F0[rows]; zz=z0[rows]
    # Use explicit pairwise row/direction extraction; advanced-index axis
    # ordering differs across otherwise similar indexing expressions.
    cc=np.stack([C[row, :, direction] for row,direction in zip(rows,dirs)])
    ss0=s0[rows]; bb=sig0[rows]
    started=time.monotonic(); a.output.mkdir(parents=True)
    progress=a.output/'progress.jsonl'
    def writeprogress(stage,**kw):
        with progress.open('a') as f:
            f.write(json.dumps({'stage':stage,'elapsed_seconds':time.monotonic()-started,**kw},allow_nan=False)+'\n');f.flush()
    writeprogress('bound',source_sha256=sha(a.source_reference),capture_sha256=sha(a.capture),
        mp_arrays_sha256=sha(a.mp_arrays),run018_arrays_sha256=sha(a.run018/'nine-basis-fd-arrays.npz'),
        selected_row_directions=len(miss),epsilon_exponents=list(GRID))
    allrecords=[]
    for power in GRID:
        epsilon=math.pow(2.0,-power)
        fplus=ff.copy(); fminus=ff.copy()
        for k,d in enumerate(dirs):
            i,j=divmod(int(d),3); fplus[k,i,j]+=epsilon; fminus[k,i,j]-=epsilon
        dp=np.empty(len(miss)); dm=np.empty(len(miss))
        for k,d in enumerate(dirs):
            i,j=divmod(int(d),3)
            dp[k]=fplus[k,i,j]-ff[k,i,j]; dm[k]=ff[k,i,j]-fminus[k,i,j]
        fs=np.concatenate((fplus,fminus)); ps=np.concatenate((pp,pp)); os=np.concatenate((oo,oo))
        zzp,itp,rp,cp,okp,why,_=project(base,ref,ps,fs,os)
        n=len(miss); zzm=zzp[n:]; it_m=itp[n:]; r_m=rp[n:]; c_m=cp[n:]; okm=okp[n:]; why_m=why[n:]
        zzp=zzp[:n]; itp=itp[:n]; rp=rp[:n]; cp=cp[:n]; okp=okp[:n]; why_p=why[:n]
        sp=np.full((n,9),np.nan); sm=np.full((n,9),np.nan)
        sigp=np.full((n,3),99.0); sigm=np.full((n,3),99.0)
        if np.any(okp):
            ids=np.flatnonzero(okp); sp[ids]=base.stress(pp[ids],fplus[ids],zzp[ids]).reshape(len(ids),9)
            sigp[ids]=signature(v2,pp[ids],oo[ids],zzp[ids])
        if np.any(okm):
            ids=np.flatnonzero(okm); sm[ids]=base.stress(pp[ids],fminus[ids],zzm[ids]).reshape(len(ids),9)
            sigm[ids]=signature(v2,pp[ids],oo[ids],zzm[ids])
        for k,(d,row,tet) in enumerate(miss):
            i,j=divmod(d,3); col=cc[k]
            samep=bool(okp[k] and np.array_equal(sigp[k],bb[k])); samem=bool(okm[k] and np.array_equal(sigm[k],bb[k]))
            cp_rel=cp_abs=cm_rel=cm_abs=cent_rel=cent_abs=None
            if samep and dp[k]!=0.0: cp_rel,cp_abs,_=metric((sp[k]-ss0[k])/dp[k],col)
            if samem and dm[k]!=0.0: cm_rel,cm_abs,_=metric((ss0[k]-sm[k])/dm[k],col)
            if samep and samem and dp[k]!=0.0 and dm[k]!=0.0:
                cent_rel,cent_abs,_=metric((sp[k]-sm[k])/(dp[k]+dm[k]),col)
            rec={
                'direction_index':d,'F_component':[i,j],'paper_row':row,'tet':tet,
                'epsilon_power':power,'epsilon':epsilon,
                'plus_strict':bool(okp[k]),'minus_strict':bool(okm[k]),
                'plus_reason':str(why_p[k]),'minus_reason':str(why_m[k]),
                'plus_residual':scalar(rp[k]),'minus_residual':scalar(r_m[k]),
                'plus_correction':scalar(cp[k]),'minus_correction':scalar(c_m[k]),
                'plus_iterations':int(itp[k]),'minus_iterations':int(it_m[k]),
                'center_branch':[float(x) for x in bb[k]],
                'plus_branch':[float(x) for x in sigp[k]],'minus_branch':[float(x) for x in sigm[k]],
                'plus_same_center_branch':samep,'minus_same_center_branch':samem,
                'realized_plus_delta':scalar(dp[k]),'realized_minus_delta':scalar(dm[k]),
                'F_center_component_hex':float(ff[k,i,j]).hex(),
                'F_plus_component_hex':float(fplus[k,i,j]).hex(),
                'F_minus_component_hex':float(fminus[k,i,j]).hex(),
                'centered_relative_l2':cent_rel,'centered_absolute_l2':cent_abs,
                'plus_one_sided_relative_l2':cp_rel,'plus_one_sided_absolute_l2':cp_abs,
                'minus_one_sided_relative_l2':cm_rel,'minus_one_sided_absolute_l2':cm_abs,
                'centered_pass_1e_6':bool(cent_rel is not None and cent_rel<=DERIVATIVE_GATE),
                'interpretation':'selected diagnostic point only; strict gates unchanged; not a native noise-floor measurement'
            }
            allrecords.append(rec)
        writeprogress('epsilon_complete',epsilon_power=power,rows=len(miss),
            plus_strict=int(okp.sum()),minus_strict=int(okm.sum()),
            branch_preserving_both=int(sum(r['plus_same_center_branch'] and r['minus_same_center_branch'] for r in allrecords[-n:])),
            centered_pass=int(sum(r['centered_pass_1e_6'] for r in allrecords[-n:])))
    with (a.output/'dense-grid-records.jsonl').open('w') as f:
        for row in allrecords: f.write(json.dumps(row,allow_nan=False)+'\n')
    per=[]
    for d,row,tet in miss:
        cases=[r for r in allrecords if r['direction_index']==d and r['paper_row']==row]
        vals=[r for r in cases if r['centered_relative_l2'] is not None]
        best=min(vals,key=lambda r:r['centered_relative_l2']) if vals else None
        passes=[r['epsilon_power'] for r in cases if r['centered_pass_1e_6']]
        adj=any(b-a==0.25 for a,b in zip(passes,passes[1:]))
        per.append({'direction_index':d,'F_component':[d//3,d%3],'paper_row':row,'tet':tet,
            'grid_points':len(cases),'strict_centered_points':len(vals),
            'centered_pass_points':passes,'adjacent_quarter_grid_pass':adj,
            'minimum_relative_error':best['centered_relative_l2'] if best else None,
            'minimum_error_power':best['epsilon_power'] if best else None,
            'minimum_absolute_error':best['centered_absolute_l2'] if best else None,
            'reference_action_l2':float(np.linalg.norm(cc[miss.index((d,row,tet))])),
            'status':'focused post-hoc grid; not a substitute for full predeclared nine-basis qualification'})
    report={
        'schema':'numi.cardboard.bilinear15.nine-basis-dense-unresolved-followup.v1',
        'classification':'CPU-only selected row/direction FP64 diagnostic; no native or physical claim',
        'source173_sha256':sha(a.source_reference),'capture_sha256':sha(a.capture),'mp_arrays_sha256':sha(a.mp_arrays),
        'run018_arrays_sha256':sha(a.run018/'nine-basis-fd-arrays.npz'),
        'run018_report_sha256':sha(a.run018/'report.json'),
        'selected_cases':len(miss),'epsilon_powers':list(GRID),'per_case':per,
        'strict_gates_unchanged':{'scaled_residual':STRICT_R,'dynamic_scaled_correction':STRICT_C,'centered_relative_l2':DERIVATIVE_GATE},
        'detail_file':'dense-grid-records.jsonl',
        'limitations':['Cases were selected because they lacked adjacent passes in run018; this targeted grid cannot promote the full nine-basis gate.',
            'Branch transitions and failed endpoints are retained with status and reason.',
            'No absolute numerical noise floor is claimed; repeated deterministic FP64 evaluation is not a measured floor.'],
        'elapsed_seconds':time.monotonic()-started}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'records':len(allrecords),'output':str(a.output),'elapsed':report['elapsed_seconds']}))
if __name__=='__main__': main()
