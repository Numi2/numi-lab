#!/usr/bin/env python3
"""Exact-local flip search that permits explicitly unioned source-patch seams.

It works only on the declared 9570+9577 and 10226 patches. Patch outer edges
and boundaries between those groups remain locked. Cross-source faces get
multi-source/detail/subface/immediate-parent CSR lineage; no false single
parent is assigned. Construction only; exact whole-surface and bone gates are
still required.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OWNER402=HERE.parent/'local-two-parent-retriangulation-402'/'two_parent_patch_flips_402.py'
SEARCH402=HERE.parent/'local-two-parent-retriangulation-402'/'exact_local_flip_search_402.py'
PRED=Path('/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/source-route-weighting-owner-385/validation/src/numilab_human/cardiac_cavity_intersections.py')
GEOM=Path('/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022-10-22/wrist-connected-repair-001/incremental-exact-audit-owner-253/src')
DETAIL_GROUPS=((9570,9577),(10226,))

def load_mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def csr_rows(ptr, vals, n):
    ptr=np.asarray(ptr);vals=np.asarray(vals)
    if ptr.shape!=(n+1,) or int(ptr[0])!=0 or int(ptr[-1])!=len(vals) or np.any(ptr[1:]<ptr[:-1]):raise ValueError('invalid CSR')
    return [list(map(int,vals[int(ptr[i]):int(ptr[i+1])])) for i in range(n)]

def csr_build(rows,dtype=np.int64):
    ptr=np.zeros(len(rows)+1,dtype=np.int64); vals=[]
    for i,r in enumerate(rows): vals.extend(sorted(set(map(int,r))));ptr[i+1]=len(vals)
    return ptr,np.asarray(vals,dtype=dtype)

def detail_csr(c,patch):
    n=len(c['source_O_faces'])
    if 'face_fine_detail_ancestry_indptr' in c:
        return csr_rows(c['face_fine_detail_ancestry_indptr'],c['face_fine_detail_ancestry_rows'],n)
    out=[]
    src=csr_rows(c['face_source_ancestry_indptr'],c['face_source_ancestry_face_rows'],n)
    for i,d in enumerate(c['parent_fine_detail_face_row']):
        if int(d)>=0: out.append([int(d)])
        else: out.append(sorted(set(src[i])&set(patch)))
    return out

def group_of(ds):
    ds=set(map(int,ds))
    if ds and ds.issubset({9570,9577}): return 0
    if ds and ds.issubset({10226}): return 1
    return None

def selected_rows(c,detail_rows):
    drows=detail_csr(c,detail_rows)
    return np.asarray([i for i,r in enumerate(drows) if group_of(r) is not None],dtype=np.int64),drows

def group_scope(faces,selected):
    nodes=np.unique(np.asarray(faces)[selected])
    return np.flatnonzero(np.any(np.isin(faces,nodes),axis=1))

def boundary_locks(c,drows,patch):
    b=load_mod('two_parent_patch_flips_402',OWNER402)
    faces=c['source_O_faces']; em=b.edge_map(faces); selected=[i for i,r in enumerate(drows) if group_of(r) is not None]
    outer=set();between=set()
    for e,inc in em.items():
        if len(inc)>2: raise ValueError(f'nonmanifold edge {e}')
        if len(inc)==1:
            if group_of(drows[inc[0][0]]) is not None:outer.add(e)
        else:
            g0=group_of(drows[inc[0][0]]);g1=group_of(drows[inc[1][0]])
            if (g0 is None)!=(g1 is None):outer.add(e)
            elif g0 is not None and g1 is not None and g0!=g1:between.add(e)
    return outer|between,outer,between,selected

def group_local_audit(exact,x,f,scope):
    rec,_=exact.float32_surface_records(np.asarray(x,dtype=np.float32),f,source_seam_groups=())
    r=exact._audit_self_pairs_touching_faces(rec,set(map(int,scope)))
    pairs=np.asarray(r.get('triangle_pairs',()),dtype=np.int64).reshape((-1,2))
    return r,pairs

def propose_one(c,edge,inc,locked,min_gain=-180.0):
    b=load_mod('two_parent_patch_flips_402',OWNER402)
    f=c['source_O_faces'];d=c['parent_fine_detail_face_row'];s=c['parent_fine_subface_row']
    i,j=map(lambda z:int(z[0]),inc)
    drows=detail_csr(c,set((9570,9577,10226)))
    gi,gj=group_of(drows[i]),group_of(drows[j])
    if gi is None or gi!=gj:return None,'different_or_unselected_connected_patch'
    if edge in locked:return None,'locked_outer_or_between_patch_boundary'
    if int(c['face_kind'][i])!=int(c['face_kind'][j]):return None,'material_parent_mismatch'
    # Existing 402 geometric guard knows the 2->2 orientation, convexity and both
    # current/037 area checks. Temporarily equalize only singular source labels
    # for its eligibility check; actual lineage is merged explicitly at apply.
    temp=dict(c);temp['parent_fine_detail_face_row']=np.array(d,copy=True);temp['parent_fine_subface_row']=np.array(s,copy=True)
    du=sorted(set(drows[i])|set(drows[j]))
    srcsub=csr_rows(c['face_fine_subface_ancestry_indptr'],c['face_fine_subface_ancestry_rows'],len(f))
    su=sorted(set(srcsub[i])|set(srcsub[j]))
    common_d=du[0] if du else int(d[i]);common_s=su[0] if su else int(s[i])
    temp['parent_fine_detail_face_row'][[i,j]]=common_d;temp['parent_fine_subface_row'][[i,j]]=common_s
    em=b.edge_map(f)
    op,why=b.candidate_flip(temp,edge,inc,{common_d},locked,min_gain,0.0,em)
    if op is None:return None,why
    if 'refinement_face_parent_ancestry_indptr' in c:
        pr=csr_rows(c['refinement_face_parent_ancestry_indptr'],c['refinement_face_parent_ancestry_face_rows'],len(f))
        pu=sorted(set(pr[i])|set(pr[j]))
    else: pu=sorted(set(map(int,c['refinement_face_parent_face_row'][[i,j]]))-{ -1 })
    op=dict(op);op.update({'detail_face_union':du,'subface_union':su,'immediate_parent_union':pu,'connected_patch_group':int(gi),'cross_detail':bool(len(du)>1)})
    return op,'candidate'

def apply_op(c,op):
    b=load_mod('two_parent_patch_flips_402',OWNER402)
    out={k:np.array(v,copy=True) for k,v in c.items()};i,j=map(int,op['face_rows'])
    out['source_O_faces'][i]=op['new_faces'][0];out['source_O_faces'][j]=op['new_faces'][1]
    out['area_reference_037_source_O_faces'][i]=op['new_faces'][0];out['area_reference_037_source_O_faces'][j]=op['new_faces'][1]
    n=len(out['source_O_faces'])
    src=csr_rows(out['face_source_ancestry_indptr'],out['face_source_ancestry_face_rows'],n)
    sub=csr_rows(out['face_fine_subface_ancestry_indptr'],out['face_fine_subface_ancestry_rows'],n)
    par=csr_rows(out['refinement_face_parent_ancestry_indptr'],out['refinement_face_parent_ancestry_face_rows'],n) if 'refinement_face_parent_ancestry_indptr' in out else [[int(v)] if int(v)>=0 else [] for v in out['refinement_face_parent_face_row']]
    det=detail_csr(out,{9570,9577,10226})
    ds=sorted(set(det[i])|set(det[j]));ss=sorted(set(sub[i])|set(sub[j]));ps=sorted(set(par[i])|set(par[j]));fs=sorted(set(src[i])|set(src[j]))
    for r in (i,j):src[r]=fs;sub[r]=ss;par[r]=ps;det[r]=ds
    out['face_source_ancestry_indptr'],out['face_source_ancestry_face_rows']=csr_build(src,np.asarray(c['face_source_ancestry_face_rows']).dtype)
    out['face_fine_subface_ancestry_indptr'],out['face_fine_subface_ancestry_rows']=csr_build(sub,np.asarray(c['face_fine_subface_ancestry_rows']).dtype)
    out['refinement_face_parent_ancestry_indptr'],out['refinement_face_parent_ancestry_face_rows']=csr_build(par,np.int64)
    out['face_fine_detail_ancestry_indptr'],out['face_fine_detail_ancestry_rows']=csr_build(det,np.int64)
    out['refinement_face_parent_ancestry_semantics']=np.asarray(['CSR union of immediate parent face rows; singular row -1 and singular barycentrics NaN when multiple parents apply'])
    out['face_fine_detail_ancestry_semantics']=np.asarray(['CSR union of source fine detail face rows; singular row -1 for cross-detail flips'])
    out['refinement_face_parent_barycentric_semantics']=np.asarray(['single-parent affine map only; NaN explicitly marks multi-parent patch flips'])
    out['refinement_face_parent_face_row'][[i,j]]=ps[0] if len(ps)==1 else -1
    if len(ps)>1:out['refinement_face_parent_barycentric'][[i,j]]=np.nan
    out['parent_fine_detail_face_row'][[i,j]]=ds[0] if len(ds)==1 else -1
    out['parent_fine_subface_row'][[i,j]]=ss[0] if len(ss)==1 else -1
    for key in ('parent_coarse_face_row','parent_source_face_marker'):
        if key in out and int(out[key][i])!=int(out[key][j]):out[key][[i,j]]=-1
    return out

def audit_group(exact,c,group):
    drows=detail_csr(c,{9570,9577,10226});selected=np.asarray([i for i,r in enumerate(drows) if group_of(r)==group],dtype=np.int64)
    scope=group_scope(c['source_O_faces'],selected)
    report,pairs=group_local_audit(exact,c['points_world_m'],c['source_O_faces'],scope)
    return selected,scope,report,pairs

def search(candidate_path,output_dir,max_trials=64,max_accept=8):
    b=load_mod('two_parent_patch_flips_402',OWNER402)
    sys.path.insert(0,str(HERE));s=load_mod('exact_local_flip_search_402',SEARCH402);s.PRED=PRED
    exact=s.exact_owner()
    cp=Path(candidate_path);od=Path(output_dir)
    if od.exists():raise FileExistsError(od)
    od.mkdir(parents=True)
    with np.load(cp,allow_pickle=False) as z:c={k:z[k] for k in z.files}
    b.validate_candidate(c)
    out={k:np.array(v,copy=True) for k,v in c.items()}
    locks,outer,between,_=boundary_locks(out,detail_csr(out,{9570,9577,10226}),{9570,9577,10226})
    baseline_groups={};alltrial=[];accepted=[]
    for group in (0,1):
        selected,scope,ar,pairs=audit_group(exact,out,group)
        baseline_groups[str(group)]={'initial_pair_count':len(pairs),'initial_pairs':pairs.tolist(),'selected_face_count':len(selected),'local_scope_face_count':len(scope)}
    locks,outer,between,_=boundary_locks(out,detail_csr(out,{9570,9577,10226}),{9570,9577,10226})
    # First exercise the formerly locked 9570<->9577 source seam. When its
    # baseline local pair count is zero, accept only a zero-pair result.
    group=0
    _,_,_,pairs=audit_group(exact,out,group)
    dr=detail_csr(out,{9570,9577,10226});em=b.edge_map(out['source_O_faces']);edges=[]
    for e,inc in em.items():
        if len(inc)==2:
            da,db=set(dr[inc[0][0]]),set(dr[inc[1][0]])
            if da and db and da!=db and da.issubset({9570,9577}) and db.issubset({9570,9577}):edges.append((e,inc))
    best=None;tested=0
    for edge,inc in sorted(edges):
        if tested>=max_trials:break
        locks,_,_,_=boundary_locks(out,detail_csr(out,{9570,9577,10226}),{9570,9577,10226})
        op,why=propose_one(out,edge,inc,locks)
        if op is None:
            alltrial.append({'group':group,'edge':list(edge),'status':'REJECTED_GEOMETRY_OR_SCOPE','reason':why});continue
        tested+=1;trial=apply_op(out,op)
        sel2,scope2,arep,pp=audit_group(exact,trial,group)
        before=set(map(tuple,pairs.tolist()));after=set(map(tuple,pp.tolist()))
        safe=(len(pp)==0 if len(pairs)==0 else len(pp)<len(pairs))
        rec={'group':group,'edge':list(edge),'face_rows':op['face_rows'],'expected_old_faces':out['source_O_faces'][op['face_rows']].tolist(),'new_faces':op['new_faces'],
             'detail_face_union':op['detail_face_union'],'subface_union':op['subface_union'],'immediate_parent_union':op['immediate_parent_union'],
             'cross_detail':op['cross_detail'],'before_pairs':len(pairs),'after_pairs':len(pp),'introduced_pairs':sorted([list(v) for v in after-before]),
             'removed_pairs':sorted([list(v) for v in before-after]),'angle_gain_deg':op['angle_gain_deg'],'exact_scope_face_count':len(scope2),'exact_predicate_complete_face_count':int(arep.get('complete_surface_face_count',-1))}
        rec['status']='ACCEPTED' if safe else 'REJECTED_EXACT_LOCAL'
        alltrial.append(rec)
        if safe:
            score=(len(pp),-float(op['angle_gain_deg']),edge)
            if best is None or score<best[0]:best=(score,trial,op,pp,rec)
    if best:
        _,out,op,pairs,rec=best;accepted.append(rec)
        locks2,outer2,between2,_=boundary_locks(out,detail_csr(out,{9570,9577,10226}),{9570,9577,10226})
        if outer2!=outer or between2!=between:raise RuntimeError('cross-seam flip changed fixed patch outer/group boundary')
    # Continue exact-pair-guided flips in connected 10226 patch. Each round
    # rebuilds its seeds and all incident edges from the accepted topology.
    group=1
    for _round in range(max_accept):
        selected,scope,ar,pairs=audit_group(exact,out,group)
        if not len(pairs):break
        seeds=set(map(int,pairs.ravel())) & set(map(int,selected))
        em=b.edge_map(out['source_O_faces']);dedup={}
        for seed in sorted(seeds):
            for a0,b0 in zip(out['source_O_faces'][seed],np.roll(out['source_O_faces'][seed],-1)):
                e=(min(int(a0),int(b0)),max(int(a0),int(b0)))
                if e in em and e not in dedup:dedup[e]=em[e]
        best=None;tested=0
        for edge,inc in sorted(dedup.items()):
            if tested>=max_trials:break
            locks,_,_,_=boundary_locks(out,detail_csr(out,{9570,9577,10226}),{9570,9577,10226})
            op,why=propose_one(out,edge,inc,locks)
            if op is None:
                alltrial.append({'group':group,'edge':list(edge),'status':'REJECTED_GEOMETRY_OR_SCOPE','reason':why});continue
            tested+=1;trial=apply_op(out,op)
            sel2,scope2,arep,pp=audit_group(exact,trial,group)
            before=set(map(tuple,pairs.tolist()));after=set(map(tuple,pp.tolist()))
            safe=len(pp)<len(pairs)
            rec={'group':group,'edge':list(edge),'face_rows':op['face_rows'],'expected_old_faces':out['source_O_faces'][op['face_rows']].tolist(),'new_faces':op['new_faces'],
                 'detail_face_union':op['detail_face_union'],'subface_union':op['subface_union'],'immediate_parent_union':op['immediate_parent_union'],
                 'cross_detail':op['cross_detail'],'before_pairs':len(pairs),'after_pairs':len(pp),'introduced_pairs':sorted([list(v) for v in after-before]),
                 'removed_pairs':sorted([list(v) for v in before-after]),'angle_gain_deg':op['angle_gain_deg'],'exact_scope_face_count':len(scope2),'exact_predicate_complete_face_count':int(arep.get('complete_surface_face_count',-1))}
            rec['status']='ACCEPTED' if safe else 'REJECTED_EXACT_LOCAL'
            alltrial.append(rec)
            if safe:
                score=(len(pp),-float(op['angle_gain_deg']),edge)
                if best is None or score<best[0]:best=(score,trial,op,pp,rec)
        if best is None:break
        _,out,op,pairs,rec=best;accepted.append(rec)
        locks2,outer2,between2,_=boundary_locks(out,detail_csr(out,{9570,9577,10226}),{9570,9577,10226})
        if outer2!=outer or between2!=between:raise RuntimeError('local exact-pair flip changed fixed patch outer/group boundary')
    # Final reports per group on final composed topology.
    final_groups={}
    for group in (0,1):
        sel,scope,ar,pairs=audit_group(exact,out,group)
        final_groups[str(group)]={'final_pair_count':len(pairs),'final_pairs':pairs.tolist(),'selected_face_count':len(sel),'local_scope_face_count':len(scope),
                                  'exact_predicate_selected_face_count':int(ar.get('selected_face_count',-1)),'exact_predicate_complete_surface_face_count':int(ar.get('complete_surface_face_count',-1))}
    outp=od/'candidate.npz';np.savez_compressed(outp,**out)
    report={'status':'CROSS_SEAM_LOCAL_SEARCH_COMPLETE','geometry_admitted':False,'accepted_physical_steps':0,
            'candidate_input_path':str(cp),'candidate_input_sha256':sha_file(cp),'candidate_output_path':str(outp),'candidate_output_sha256':sha_file(outp),
            'owner_402_sha256':sha_file(OWNER402),'builder_sha256':sha_file(Path(__file__)),'predicate_owner_sha256':sha_file(PRED),
            'source_detail_patch_groups':[list(x) for x in DETAIL_GROUPS],
            'initial_groups':baseline_groups,'final_groups':final_groups,'accepted_operations':accepted,'trial_count':len(alltrial),'trials':alltrial,
            'fixed_outer_edge_count':len(outer),'fixed_outer_edges_preserved':outer==boundary_locks(out,detail_csr(out,{9570,9577,10226}),{9570,9577,10226})[1],
            'fixed_between_group_boundary_count':len(between),'fixed_between_group_boundary_preserved':between==boundary_locks(out,detail_csr(out,{9570,9577,10226}),{9570,9577,10226})[2],
            'source_patch_seam_policy':'9570<->9577 interior seam is retriangulable; only combined patch outer ring and boundaries to unrelated selected groups are fixed',
            'points_bitwise_unchanged':b.array_identical(c['points_world_m'],out['points_world_m']),
            'matched_037_faces_synchronized':bool(np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces'])),
            'multi_source_parent_unions_explicit':bool('face_fine_detail_ancestry_indptr' in out and 'refinement_face_parent_ancestry_indptr' in out),
            'full_surface_and_all185_gates_required':True,
            'scope_limit':'Exact self predicate only on each declared source-detail group plus its vertex one-ring; not a full-surface or bone audit.'}
    (od/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--max-trials',type=int,default=64);ap.add_argument('--max-accept',type=int,default=8)
    a=ap.parse_args();r=search(a.candidate,a.output_dir,a.max_trials,a.max_accept)
    print(json.dumps({k:v for k,v in r.items() if k not in ('trials','accepted_operations')},indent=2))
if __name__=='__main__':main()
