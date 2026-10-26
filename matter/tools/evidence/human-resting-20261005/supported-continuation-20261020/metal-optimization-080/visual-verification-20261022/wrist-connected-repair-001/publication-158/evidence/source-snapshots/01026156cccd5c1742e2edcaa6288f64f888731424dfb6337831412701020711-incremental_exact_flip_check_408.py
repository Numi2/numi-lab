from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class State:
    owner: object
    points: np.ndarray
    faces: np.ndarray
    records: tuple
    index: object
    pairs: frozenset
    seams: tuple
    candidate_sha256: str
    predicate_sha256: str

@dataclass(frozen=True)
class Trial:
    changed: tuple
    replacements: tuple
    faces: np.ndarray
    pairs: frozenset
    report: dict
    state_id: int

def _pairs(a,n):
    a=np.asarray(a)
    # The pinned predicate emits [] (shape (0,), float dtype) for an empty
    # result; accept that one canonical empty representation as well as an
    # explicitly typed (0,2) table. Other malformed empty shapes still fail.
    if a.size==0:
        if a.ndim==1 and a.shape==(0,):
            return frozenset()
        if a.ndim==2 and a.shape==(0,2) and np.issubdtype(a.dtype,np.integer):
            return frozenset()
        raise ValueError("pairs must be integer Nx2")
    if a.ndim!=2 or a.shape[1]!=2 or not np.issubdtype(a.dtype,np.integer): raise ValueError("pairs must be integer Nx2")
    s=set()
    for u,v in a:
        u,v=int(u),int(v)
        if u<0 or v<0 or u>=n or v>=n or u==v: raise ValueError("invalid pair face IDs")
        s.add((min(u,v),max(u,v)))
    if len(s)!=len(a): raise ValueError("duplicate pair")
    return frozenset(s)

def prepare(owner,points,faces,pair_rows,*,audit_report,candidate_sha256,predicate_sha256,expected_predicate_sha256,source_seam_groups=()):
    x,f=np.asarray(points),np.asarray(faces)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all(): raise ValueError("points must be finite Float32 Nx3")
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer): raise ValueError("faces must be integer Fx3")
    if f.size and (f.min()<0 or f.max()>=len(x)): raise ValueError("face index out of range")
    if not isinstance(audit_report,dict) or audit_report.get("complete_surface_audit") is not True: raise ValueError("baseline audit is incomplete")
    if audit_report.get("candidate_sha256")!=candidate_sha256: raise ValueError("candidate pin mismatch")
    if predicate_sha256!=expected_predicate_sha256 or audit_report.get("surface_predicate_sha256")!=expected_predicate_sha256: raise ValueError("predicate pin mismatch")
    if int(audit_report.get("faces",-1))!=len(f): raise ValueError("face count mismatch")
    seams=tuple(tuple(map(int,g)) for g in source_seam_groups)
    expected=_pairs(pair_rows,len(f))
    if int(audit_report.get("exact_PLC_surface_pairs",-1))!=len(expected): raise ValueError("pair count mismatch")
    records,_=owner.float32_surface_records(x,f,source_seam_groups=seams)
    records=owner._immutable_self_pair_records(records)
    if len(records)!=len(f) or tuple(r[3] for r in records)!=tuple(range(len(f))): raise ValueError("baseline records do not preserve face IDs")
    full=owner._audit_self_pairs_touching_faces(records,set(range(len(f))))
    if _pairs(full["triangle_pairs"],len(f))!=expected: raise ValueError("baseline pair array disagrees with exact predicate replay")
    index=owner._prepare_surface_aabb(records)
    xx,ff=np.array(x,copy=True),np.array(f,copy=True);xx.setflags(write=False);ff.setflags(write=False)
    return State(owner,xx,ff,records,index,expected,seams,str(candidate_sha256),str(predicate_sha256))

def prepare_full(owner,points,faces,*,candidate_sha256,predicate_sha256,expected_predicate_sha256,source_seam_groups=()):
    """Build the exact full baseline once and return its reusable AABB state."""
    x,f=np.asarray(points),np.asarray(faces)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all(): raise ValueError("points must be finite Float32 Nx3")
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer): raise ValueError("faces must be integer Fx3")
    if f.size and (f.min()<0 or f.max()>=len(x)): raise ValueError("face index out of range")
    if predicate_sha256!=expected_predicate_sha256: raise ValueError("predicate pin mismatch")
    seams=tuple(tuple(map(int,g)) for g in source_seam_groups)
    records,_=owner.float32_surface_records(x,f,source_seam_groups=seams)
    records=owner._immutable_self_pair_records(records)
    if len(records)!=len(f) or tuple(r[3] for r in records)!=tuple(range(len(f))): raise ValueError("baseline records do not preserve face IDs")
    audit=owner._audit_self_pairs_touching_faces(records,set(range(len(f))))
    if int(audit.get("selected_face_count",-1))!=len(f) or int(audit.get("complete_surface_face_count",-1))!=len(f):
        raise ValueError("exact owner did not audit every baseline face")
    pairs=_pairs(audit.get("triangle_pairs",[]),len(f))
    if int(audit.get("exact_PLC_surface_pairs",len(pairs)))!=len(pairs): raise ValueError("exact owner pair count is inconsistent")
    report={"complete_surface_audit":True,"candidate_sha256":str(candidate_sha256),
            "surface_predicate_sha256":str(predicate_sha256),"faces":len(f),
            "exact_PLC_surface_pairs":len(pairs),"selected_face_count":len(f),
            "complete_surface_face_count":len(f)}
    index=owner._prepare_surface_aabb(records)
    xx,ff=np.array(x,copy=True),np.array(f,copy=True);xx.setflags(write=False);ff.setflags(write=False)
    state=State(owner,xx,ff,records,index,pairs,seams,str(candidate_sha256),str(predicate_sha256))
    return state,report,audit

def evaluate(state,points,trial_faces,*,changed_face_rows=None):
    """Reuse only unchanged-face pairs; exact-test changed rows against full cached baseline AABB."""
    x,f=np.asarray(points),np.asarray(trial_faces)
    if x.dtype!=np.float32 or x.shape!=state.points.shape or x.tobytes()!=state.points.tobytes(): raise ValueError("coordinates must be bitwise-identical Float32 baseline")
    if f.dtype!=state.faces.dtype or f.shape!=state.faces.shape or not np.issubdtype(f.dtype,np.integer): raise ValueError("trial face shape/dtype differs")
    if f.size and (f.min()<0 or f.max()>=len(x)): raise ValueError("trial face index out of range")
    changed=tuple(map(int,np.flatnonzero(np.any(f!=state.faces,axis=1))))
    if changed_face_rows is not None and tuple(sorted(set(map(int,changed_face_rows))))!=changed: raise ValueError("declared changed rows do not match face delta")
    if not changed: raise ValueError("trial has no changed faces")
    local,_=state.owner.float32_surface_records(x,f[np.asarray(changed,dtype=np.int64)],source_seam_groups=state.seams)
    repl=tuple((r[0],r[1],r[2],row,r[4]) for r,row in zip(local,changed))
    cset=set(changed);cand=[];seen=set()
    for new in repl:
        for _,old in state.owner._query_surface_aabb((new,),state.index):
            i,j=int(new[3]),int(old[3])
            if j in cset or i==j: continue
            key=(min(i,j),max(i,j))
            if key not in seen:
                seen.add(key);cand.append((new,old) if i<j else (old,new))
    if len(repl)>1:
        for a,b in state.owner._aabb_candidate_pairs(repl,repl,same_surface=True):
            key=(min(int(a[3]),int(b[3])),max(int(a[3]),int(b[3])))
            if key not in seen: seen.add(key);cand.append((a,b))
    checked=state.owner._audit_record_pairs(cand,same_surface=True)
    changed_pairs=frozenset(tuple(map(int,p)) for p in checked["triangle_pairs"])
    retained=frozenset(p for p in state.pairs if p[0] not in cset and p[1] not in cset)
    after=retained|changed_pairs
    introduced,removed=sorted(after-state.pairs),sorted(state.pairs-after)
    rep={"status":"EXACT_INCREMENTAL_GLOBAL_SELF_AUDIT","complete_surface_audit":True,"exact_incremental_reuse":True,
         "candidate_sha256":state.candidate_sha256,"surface_predicate_sha256":state.predicate_sha256,"faces":len(state.faces),
         "changed_face_rows":list(changed),"unchanged_face_count":len(state.faces)-len(changed),
         "unchanged_baseline_pairs_reused":len(retained),"changed_face_aabb_candidate_pairs":int(checked["aabb_candidate_pairs"]),
         "changed_face_exact_pairs":len(changed_pairs),"allowed_shared_vertex_or_edge_pairs":int(checked["allowed_shared_vertex_or_edge_pairs"]),
         "exact_PLC_surface_pairs":len(after),"introduced_pairs":[list(p) for p in introduced],"removed_pairs":[list(p) for p in removed],
         "no_new_pairs":not introduced,"strict_pair_reduction":len(after)<len(state.pairs),
         "scope_limit":"Global exact result: unchanged pairs reuse complete baseline; every pair touching changed faces is re-audited against the full surface."}
    ff=np.array(f,copy=True);ff.setflags(write=False)
    return Trial(changed,repl,ff,after,rep,id(state))

def commit(state,trial,*,require_reduction=True,candidate_sha256=None):
    if trial.state_id!=id(state): raise ValueError("trial belongs to another baseline")
    if trial.report["introduced_pairs"]: raise ValueError("trial introduces exact self-pairs")
    if require_reduction and len(trial.pairs)>=len(state.pairs): raise ValueError("trial does not reduce exact self-pairs")
    records=list(state.records)
    for r in trial.replacements: records[int(r[3])]=r
    records=state.owner._immutable_self_pair_records(records)
    index=state.owner._prepare_surface_aabb(records)
    next_sha=state.candidate_sha256 if candidate_sha256 is None else str(candidate_sha256)
    if len(next_sha)!=64 or any(c not in "0123456789abcdef" for c in next_sha.lower()): raise ValueError("candidate SHA-256 must be 64 hexadecimal characters")
    return State(state.owner,state.points,trial.faces,records,index,trial.pairs,state.seams,next_sha,state.predicate_sha256)
