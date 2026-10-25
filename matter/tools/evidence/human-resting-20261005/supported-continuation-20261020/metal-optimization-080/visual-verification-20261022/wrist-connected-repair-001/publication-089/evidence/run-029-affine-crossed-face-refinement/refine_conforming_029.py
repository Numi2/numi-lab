#!/usr/bin/env python3
"""Conforming, ancestry-preserving subdivision of exact crossed O faces."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

BASE = Path('/Users/n/Work/numi-human-connected-surface-069')
EVIDENCE = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
PREDICATE = EVIDENCE/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
EXPECTED_CANDIDATE_SHA = '2f93d0159f5e1ece1fa377da71e52e4edb7ee60f4b61e483a0390733ca404c11'
EXPECTED_FACE_REPORT_SHA = '6dd739e73313074032f6bb750aa0eb1ff00d251de94e032181345cc46ad9f0a9'


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def crossing_face_rows(face_report: dict, nfaces: int) -> tuple[np.ndarray, int]:
    if not face_report.get('complete') or face_report.get('geometry_or_native_admission') is not False:
        raise ValueError('face report must be complete and diagnostic-only')
    rows=[]; pair_count=0
    for bank in face_report.get('banks',[]):
        result=bank.get('result')
        if result is None: continue
        if int(result.get('count',-1)) != len(result.get('triangle_pairs') or []):
            raise ValueError('bank hit count does not match triangle pair rows')
        for pair in result.get('triangle_pairs') or []:
            if len(pair)!=2: raise ValueError('malformed triangle pair')
            face=int(pair[0]);
            if face<0 or face>=nfaces: raise ValueError('crossed skin face row out of range')
            rows.append(face); pair_count+=1
    unique=np.unique(np.asarray(rows,dtype=np.int64))
    if len(unique)==0: raise ValueError('no crossed surface faces')
    return unique,pair_count


def _edge_key(a:int,b:int)->tuple[int,int]:
    return (a,b) if a<b else (b,a)


def _edge_incidence(faces: np.ndarray) -> dict[tuple[int,int],list[int]]:
    out=collections.defaultdict(list)
    for fi,(a,b,c) in enumerate(np.asarray(faces,dtype=np.int64)):
        out[_edge_key(int(a),int(b))].append(fi)
        out[_edge_key(int(b),int(c))].append(fi)
        out[_edge_key(int(c),int(a))].append(fi)
    return dict(out)


def _boundary_components(edges:set[tuple[int,int]]) -> tuple[int,dict[int,int]]:
    adj=collections.defaultdict(set)
    for a,b in edges: adj[a].add(b); adj[b].add(a)
    degree={v:len(ns) for v,ns in adj.items()}
    seen=set(); comps=0
    for start in adj:
        if start in seen: continue
        comps+=1; stack=[start]; seen.add(start)
        while stack:
            u=stack.pop()
            for v in adj[u]:
                if v not in seen: seen.add(v); stack.append(v)
    return comps,degree


def _bary_for_vertex(v:int, tri:tuple[int,int,int])->np.ndarray:
    out=np.zeros(3,dtype=np.float64)
    out[tri.index(v)]=1.0
    return out


def _remap_csr(ptr:np.ndarray, vals:np.ndarray, parents:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    ptr=np.asarray(ptr,dtype=np.int64); vals=np.asarray(vals)
    if len(ptr)!=int(np.max(parents,initial=-1))+2: raise ValueError('CSR parent pointer length mismatch')
    out_ptr=[0]; out_vals=[]
    for p in parents:
        out_vals.extend(vals[ptr[int(p)]:ptr[int(p)+1]].tolist())
        out_ptr.append(len(out_vals))
    return np.asarray(out_ptr,dtype=np.int64),np.asarray(out_vals,dtype=vals.dtype)


def legacy_build_refinement(points:np.ndarray, faces:np.ndarray, crossed_faces:np.ndarray) -> dict:
    X=np.asarray(points)
    F=np.asarray(faces)
    if X.dtype!=np.float32 or X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():
        raise ValueError('points must be finite Float32 Nx3')
    if F.dtype.kind not in 'iu' or F.ndim!=2 or F.shape[1]!=3: raise ValueError('faces must be integer Mx3')
    F=F.astype(np.int64,copy=False)
    if F.min(initial=0)<0 or F.max(initial=-1)>=len(X): raise ValueError('face node out of range')
    if np.any(F[:,0]==F[:,1]) or np.any(F[:,1]==F[:,2]) or np.any(F[:,2]==F[:,0]): raise ValueError('repeated face vertex')
    if len(np.unique(np.sort(F,axis=1),axis=0))!=len(F): raise ValueError('duplicate triangle')
    crossed=np.unique(np.asarray(crossed_faces,dtype=np.int64))
    if crossed.ndim!=1 or not len(crossed) or crossed.min()<0 or crossed.max()>=len(F): raise ValueError('invalid crossed face rows')
    incidence=_edge_incidence(F)
    if any(len(v)>2 for v in incidence.values()): raise ValueError('input surface has nonmanifold edge')
    old_boundary={e for e,fs in incidence.items() if len(fs)==1}
    old_bc,old_bd=_boundary_components(old_boundary)
    if any(d!=2 for d in old_bd.values()): raise ValueError('input boundary is not a union of simple loops')

    selected_edges=set()
    for fi in crossed:
        a,b,c=map(int,F[int(fi)])
        selected_edges.update((_edge_key(a,b),_edge_key(b,c),_edge_key(c,a)))
    affected=sorted({fi for e in selected_edges for fi in incidence[e]})
    if not set(crossed).issubset(affected): raise AssertionError('crossed face missing from affected set')
    edges_sorted=sorted(selected_edges)
    edge_node={e:len(X)+i for i,e in enumerate(edges_sorted)}
    new_points=[]; node_parent_rows=[]; node_parent_weights=[]
    node_ref_kind=np.zeros(len(X),dtype=np.int8); node_parent_edges=np.full((len(X),2),-1,dtype=np.int32)
    node_parent_face_ptr=[0]; node_parent_face_rows=[]; node_parent_face_bary=[]
    node_prolong_ptr=[0]; node_prolong_cols=[]; node_prolong_weights=[]
    for i in range(len(X)):
        node_prolong_cols.append(i); node_prolong_weights.append(1.0); node_prolong_ptr.append(len(node_prolong_cols))
        node_parent_face_ptr.append(0 if i==0 else node_parent_face_ptr[-1])
    # Convert old-node parent CSR pointer to canonical zeros for every old node.
    node_parent_face_ptr=list(np.zeros(len(X)+1,dtype=np.int64))
    for e in edges_sorted:
        a,b=e; q=(X[a].astype(np.float64)+X[b].astype(np.float64))*0.5
        q32=q.astype(np.float32)
        if np.array_equal(q32,X[a]) or np.array_equal(q32,X[b]): raise ValueError(f'Float32 midpoint collapses to endpoint for edge {e}')
        new_points.append(q32); node_parent_rows.append((a,b)); node_parent_weights.append((0.5,0.5)); node_ref_kind=np.append(node_ref_kind,1)
        node_parent_edges=np.vstack((node_parent_edges,np.asarray(e,dtype=np.int32)))
        for fi in incidence[e]:
            tri=tuple(map(int,F[fi])); node_parent_face_rows.append(fi)
            node_parent_face_bary.append(0.5*_bary_for_vertex(a,tri)+0.5*_bary_for_vertex(b,tri))
        node_parent_face_ptr.append(len(node_parent_face_rows))
        node_prolong_cols.extend([a,b]);node_prolong_weights.extend([0.5,0.5]);node_prolong_ptr.append(len(node_prolong_cols))
    edge_node_end=len(X)+len(edges_sorted)

    out_faces=[]; parent_faces=[]; parent_bary=[]; child_count=np.ones(len(F),dtype=np.int32)
    face_centroid_node={}
    for fi in affected:
        tri=tuple(map(int,F[fi])); poly=[]; bary=[]
        for j in range(3):
            a,b=tri[j],tri[(j+1)%3]
            poly.append(a); bary.append(_bary_for_vertex(a,tri))
            e=_edge_key(a,b)
            if e in edge_node:
                poly.append(edge_node[e]); bary.append(0.5*_bary_for_vertex(a,tri)+0.5*_bary_for_vertex(b,tri))
        center=len(X)+len(edges_sorted)+len(face_centroid_node)
        face_centroid_node[fi]=center
        center_bary=np.full(3,1/3,dtype=np.float64)
        for j in range(len(poly)):
            k=(j+1)%len(poly)
            out_faces.append((poly[j],poly[k],center)); parent_faces.append(fi)
            parent_bary.append(np.vstack((bary[j],bary[k],center_bary)))
        child_count[fi]=len(poly)
        new_points.append(np.mean(X[np.asarray(tri)].astype(np.float64),axis=0).astype(np.float32))
        node_parent_rows.append(tri);node_parent_weights.append((1/3,1/3,1/3));node_ref_kind=np.append(node_ref_kind,2)
        node_parent_edges=np.vstack((node_parent_edges,np.asarray((-1,-1),dtype=np.int32)))
        node_parent_face_rows.append(fi);node_parent_face_bary.append(center_bary)
        node_parent_face_ptr.append(len(node_parent_face_rows))
        node_prolong_cols.extend(tri);node_prolong_weights.extend([1/3]*3);node_prolong_ptr.append(len(node_prolong_cols))

    outF=[]
    parent_out=[]
    bary_out=[]
    affected_set=set(affected)
    for fi in range(len(F)):
        if fi not in affected_set:
            tri=tuple(map(int,F[fi])); outF.append(tri); parent_out.append(fi); bary_out.append(np.eye(3,dtype=np.float64))
        else:
            # Fan rows were appended in affected-face order above; build per-parent slice in same order.
            pass
    # Reconstruct child output in input-face order for stable ancestry and parent-aligned order.
    outF=[];parent_out=[];bary_out=[]
    child_lookup=collections.defaultdict(list)
    for f,p,b in zip(out_faces,parent_faces,parent_bary): child_lookup[p].append((f,b))
    for fi in range(len(F)):
        if fi in affected_set:
            for f,b in child_lookup[fi]: outF.append(f); parent_out.append(fi); bary_out.append(b)
        else:
            outF.append(tuple(map(int,F[fi])));parent_out.append(fi);bary_out.append(np.eye(3,dtype=np.float64))
    Xout=np.vstack((X,np.asarray(new_points,dtype=np.float32)))
    Fout=np.asarray(outF,dtype=np.int32)
    parent_out=np.asarray(parent_out,dtype=np.int32); bary_out=np.asarray(bary_out,dtype=np.float64)
    if len(Xout)!=len(X)+len(edges_sorted)+len(affected): raise AssertionError('unexpected node count')
    if not np.array_equal(Xout[:len(X)].view(np.uint32),X.view(np.uint32)): raise AssertionError('existing point prefix changed')
    if np.any(Fout<0) or Fout.max()>=len(Xout): raise AssertionError('output face invalid')
    out_incidence=_edge_incidence(Fout)
    if any(len(v)>2 for v in out_incidence.values()): raise ValueError('refinement produced nonmanifold edge')
    new_boundary={e for e,fs in out_incidence.items() if len(fs)==1}
    expected_boundary=set()
    for a,b in old_boundary:
        e=(a,b)
        if e in edge_node:
            m=edge_node[e];expected_boundary.add(_edge_key(a,m));expected_boundary.add(_edge_key(m,b))
        else: expected_boundary.add(e)
    if new_boundary!=expected_boundary: raise AssertionError('boundary edge chain changed beyond selected-edge bisection')
    new_bc,new_bd=_boundary_components(new_boundary)
    if new_bc!=old_bc or any(d!=2 for d in new_bd.values()): raise AssertionError('boundary loops changed topologically')

    old_cross=np.cross(X[F[:,1]].astype(np.float64)-X[F[:,0]],X[F[:,2]].astype(np.float64)-X[F[:,0]])
    new_cross=np.cross(Xout[Fout[:,1]].astype(np.float64)-Xout[Fout[:,0]],Xout[Fout[:,2]].astype(np.float64)-Xout[Fout[:,0]])
    if np.any(np.linalg.norm(new_cross,axis=1)==0): raise ValueError('Float32 refinement produced zero-area child face')
    child_parent_cross=old_cross[parent_out]
    orient=np.einsum('ij,ij->i',new_cross,child_parent_cross)
    if np.any(orient<=0): raise ValueError('child orientation is not positive relative to its parent')
    old_area=0.5*np.linalg.norm(old_cross,axis=1)
    new_area=0.5*np.linalg.norm(new_cross,axis=1)
    sums=np.bincount(parent_out,weights=new_area,minlength=len(F))
    rel=np.abs(sums-old_area)/np.maximum(old_area,1e-300)

    # Sparse prolongation from the pre-refinement node domain to every output node.
    nout=len(Xout)
    if len(node_prolong_ptr)!=nout+1: raise AssertionError('prolongation row count mismatch')
    for row in range(nout):
        cols=node_prolong_cols[node_prolong_ptr[row]:node_prolong_ptr[row+1]]
        ws=node_prolong_weights[node_prolong_ptr[row]:node_prolong_ptr[row+1]]
        if not cols or abs(sum(ws)-1.0)>1e-14: raise AssertionError('prolongation row is not affine')
    node_edge_weights=np.full((len(Xout),2),-1.0,dtype=np.float64)
    if edges_sorted: node_edge_weights[len(X):edge_node_end]=0.5
    return {
        'points_world_m':Xout,'source_O_faces':Fout,
        'parent_face_rows':parent_out,'parent_face_barycentric':bary_out,
        'selected_edges':np.asarray(edges_sorted,dtype=np.int32),'edge_midpoint_rows':np.asarray([edge_node[e] for e in edges_sorted],dtype=np.int32),
        'affected_parent_face_rows':np.asarray(affected,dtype=np.int32),'face_centroid_rows':np.asarray([face_centroid_node[f] for f in affected],dtype=np.int32),
        'refinement_node_kind':node_ref_kind,'refinement_node_parent_edge_rows':node_parent_edges,
        'refinement_node_parent_edge_weights':node_edge_weights,
        'refinement_node_parent_face_indptr':np.asarray(node_parent_face_ptr,dtype=np.int64),
        'refinement_node_parent_face_rows':np.asarray(node_parent_face_rows,dtype=np.int32),
        'refinement_node_parent_face_barycentric':np.asarray(node_parent_face_bary,dtype=np.float64).reshape(-1,3),
        'source_prolongation_indptr':np.asarray(node_prolong_ptr,dtype=np.int64),
        'source_prolongation_node_rows':np.asarray(node_prolong_cols,dtype=np.int32),
        'source_prolongation_weights':np.asarray(node_prolong_weights,dtype=np.float64),
        'old_face_area_m2':old_area,'new_face_area_sum_m2':sums,
        'boundary_components_before':old_bc,'boundary_components_after':new_bc,
        'boundary_edges_before':len(old_boundary),'boundary_edges_after':len(new_boundary),
        'child_count_by_parent_face':child_count,
    }


def _extend_node_array(key:str,a:np.ndarray,result:dict,n_old:int)->np.ndarray:
    n_new=len(result['points_world_m'])-n_old
    mids=result['selected_edges']; affected=result['affected_parent_face_rows']; Fparent=result.get('_input_faces')
    parent_rows=[]; weights=[]
    for u,v in mids: parent_rows.append((int(u),int(v)));weights.append((.5,.5))
    for fi in affected: parent_rows.append(tuple(map(int,Fparent[int(fi)])));weights.append((1/3,1/3,1/3))
    if n_new!=len(parent_rows): raise AssertionError('node interpolation parent count mismatch')
    if a.dtype.kind=='b': return np.concatenate((a,np.zeros((n_new,)+a.shape[1:],dtype=a.dtype)),axis=0)
    if a.dtype.kind in 'iu':
        fill=np.full((n_new,)+a.shape[1:],-1,dtype=a.dtype)
        if key=='node_kind': fill[:]= -1
        return np.concatenate((a,fill),axis=0)
    if a.dtype.kind in 'f':
        if key=='node_parent_coarse_edge_parameter':
            fill=np.full((n_new,)+a.shape[1:],-1.0,dtype=a.dtype)
        else:
            rows=[]
            for ids,ws in zip(parent_rows,weights): rows.append(np.tensordot(np.asarray(ws,dtype=np.float64),a[np.asarray(ids,dtype=np.int64)].astype(np.float64),axes=(0,0)).astype(a.dtype))
            fill=np.asarray(rows,dtype=a.dtype)
        return np.concatenate((a,fill),axis=0)
    if a.dtype.kind in 'US':
        return np.concatenate((a,np.full((n_new,)+a.shape[1:],'',dtype=a.dtype)),axis=0)
    raise ValueError(f'unsupported node-aligned dtype for {key}: {a.dtype}')


def extend_candidate_arrays(source:dict,result:dict,candidate_sha:str=EXPECTED_CANDIDATE_SHA,face_report_sha:str=EXPECTED_FACE_REPORT_SHA)->dict:
    X=np.asarray(source['points_world_m']); F=np.asarray(source['source_O_faces'])
    result['_input_faces']=F
    out=dict(source)
    prior_keys = (
        'refinement_node_kind','refinement_node_parent_edge_rows','refinement_node_parent_edge_weights',
        'refinement_node_parent_face_indptr','refinement_node_parent_face_rows',
        'refinement_node_parent_face_barycentric','refinement_face_parent_face_row',
        'refinement_face_parent_barycentric','refinement_input_candidate_sha256',
        'refinement_input_face_report_sha256','refinement_node_kind_semantics',
        'refinement_face_parent_barycentric_semantics','source_prolongation_indptr',
        'source_prolongation_node_rows','source_prolongation_weights','source_prolongation_semantics',
        'refinement_boundary_semantics')
    for key in prior_keys:
        if key in source:
            out['prior_run028__'+key] = np.asarray(source[key]).copy()
    parent=result['parent_face_rows']
    for k,a in list(source.items()):
        if k in ('points_world_m','source_O_faces'): continue
        if k in ('face_source_ancestry_indptr','face_source_ancestry_face_rows','face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows'): continue
        if np.asarray(a).ndim>=1 and np.asarray(a).shape[0]==len(X):
            out[k]=_extend_node_array(k,np.asarray(a),result,len(X))
        elif np.asarray(a).ndim>=1 and np.asarray(a).shape[0]==len(F):
            out[k]=np.asarray(a)[parent]
    for ptrkey,valkey in [('face_source_ancestry_indptr','face_source_ancestry_face_rows'),('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows')]:
        if ptrkey in source and valkey in source:
            ptr,vals=_remap_csr(source[ptrkey],source[valkey],parent)
            out[ptrkey]=ptr;out[valkey]=vals
    out['points_world_m']=result['points_world_m'];out['source_O_faces']=result['source_O_faces']
    for key in ('refinement_node_kind','refinement_node_parent_edge_rows','refinement_node_parent_edge_weights','refinement_node_parent_face_indptr','refinement_node_parent_face_rows','refinement_node_parent_face_barycentric','source_prolongation_indptr','source_prolongation_node_rows','source_prolongation_weights','refinement_face_parent_face_row','refinement_face_parent_barycentric'):
        if key in result: out[key]=result[key]
    out['refinement_face_parent_face_row']=parent
    out['refinement_face_parent_barycentric']=result['parent_face_barycentric']
    out['refinement_input_candidate_sha256']=np.asarray([candidate_sha])
    out['refinement_input_face_report_sha256']=np.asarray([face_report_sha])
    out['refinement_node_kind_semantics']=np.asarray(['0=preexisting;1=selected-edge midpoint;2=affected-face centroid'])
    out['refinement_face_parent_barycentric_semantics']=np.asarray(['output child vertices in barycentric coordinates of pre-refinement parent O face'])
    out['source_prolongation_semantics']=np.asarray([f'rows map immediate pre-refinement {len(X)}-node Run028 field to refined nodes using identity, edge midpoint, or parent-face centroid affine weights'])
    out['refinement_boundary_semantics']=np.asarray(['selected boundary edges are subdivided collinearly; original boundary loop topology and outside coordinates are preserved'])
    return out


def exact_self_audit(points:np.ndarray,faces:np.ndarray)->dict:
    sys.path.insert(0,str(EVIDENCE/'incremental-exact-audit-owner-253/src'))
    spec=importlib.util.spec_from_file_location('numilab_human._refine017_predicate',PREDICATE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    records,_=module.float32_surface_records(np.asarray(points,dtype=np.float32),np.asarray(faces,dtype=np.int64),source_seam_groups=())
    return module._audit_self_pairs_touching_faces(records,set(range(len(faces))))


P084_OWNER = BASE/'publication-084/canonical/matter/tools/human_skin_fat_boundary_refinement.py'
P084_OWNER_SHA = 'eab8deda28164964051a6fa6ff346b8d94bc8bede04e896411dfdd02d46594f6'

def p084_affine_refinement(points, faces, crossed, markers):
    if sha256(P084_OWNER) != P084_OWNER_SHA:
        raise ValueError('P084 owner hash mismatch')
    spec=importlib.util.spec_from_file_location('p084_boundary_refinement_084',P084_OWNER)
    owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
    raw=owner.subdivide_crossed_faces_affine(points,faces,crossed,parent_face_markers=markers)
    X=np.asarray(points);F=np.asarray(faces);Xp=raw.points;Fp=raw.faces;parents=raw.child_parent_face_rows
    old_cross=np.cross(X[F[:,1]].astype(np.float64)-X[F[:,0]].astype(np.float64),X[F[:,2]].astype(np.float64)-X[F[:,0]].astype(np.float64))
    new_cross=np.cross(Xp[Fp[:,1]].astype(np.float64)-Xp[Fp[:,0]].astype(np.float64),Xp[Fp[:,2]].astype(np.float64)-Xp[Fp[:,0]].astype(np.float64))
    old_area=.5*np.linalg.norm(old_cross,axis=1);new_area=.5*np.linalg.norm(new_cross,axis=1)
    sums=np.bincount(parents,weights=new_area,minlength=len(F))
    return {
      'points_world_m':raw.points,'source_O_faces':raw.faces,
      'parent_face_rows':raw.child_parent_face_rows,
      'parent_face_barycentric':raw.child_parent_face_barycentric,
      'selected_edges':raw.selected_edges,'edge_midpoint_rows':raw.edge_midpoint_rows,
      'affected_parent_face_rows':raw.affected_parent_face_rows,'face_centroid_rows':raw.face_centroid_rows,
      'refinement_node_kind':raw.refinement_node_kind,
      'refinement_node_parent_edge_rows':raw.refinement_node_parent_edge_rows,
      'refinement_node_parent_edge_weights':raw.refinement_node_parent_edge_weights,
      'refinement_node_parent_face_indptr':raw.refinement_node_parent_face_indptr,
      'refinement_node_parent_face_rows':raw.refinement_node_parent_face_rows,
      'refinement_node_parent_face_barycentric':raw.refinement_node_parent_face_barycentric,
      'source_prolongation_indptr':raw.source_prolongation_indptr,
      'source_prolongation_node_rows':raw.source_prolongation_node_rows,
      'source_prolongation_weights':raw.source_prolongation_weights,
      'old_face_area_m2':old_area,'new_face_area_sum_m2':sums,
      'boundary_components_before':raw.boundary_components_before,
      'boundary_components_after':raw.boundary_components_after,
      'boundary_edges_before':raw.boundary_edges_before,
      'boundary_edges_after':raw.boundary_edges_after,
      'relative_parent_area_error':raw.relative_parent_area_error,
      'p084_parent_face_markers':raw.child_parent_face_markers,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',required=True);ap.add_argument('--face-report',required=True);ap.add_argument('--expected-candidate-sha',required=True);ap.add_argument('--expected-face-report-sha',required=True);ap.add_argument('--output',required=True);ap.add_argument('--report',required=True)
    a=ap.parse_args(); cand=Path(a.candidate); fr=Path(a.face_report); out=Path(a.output); rp=Path(a.report)
    if out.exists() or rp.exists(): raise SystemExit('refuse overwrite existing output')
    csha=sha256(cand); fsha=sha256(fr)
    if csha!=a.expected_candidate_sha or fsha!=a.expected_face_report_sha: raise SystemExit('frozen input hash mismatch')
    started=time.monotonic(); z=np.load(cand,allow_pickle=False); source={k:z[k] for k in z.files}
    F=source['source_O_faces'];X=source['points_world_m']
    report=json.loads(fr.read_text())
    if report.get('candidate_sha256')!=csha: raise SystemExit('face report candidate mismatch')
    crossed,pair_count=crossing_face_rows(report,len(F))
    result=p084_affine_refinement(X,F,crossed,source['parent_source_face_marker'])
    out_arrays=extend_candidate_arrays(source,result,csha,fsha)
    # Check source arrays remain on explicit identity prefix and all face provenance follows parent rows.
    for k in ('node_kind','node_source_row','node_source_virtual_row','node_parent_coarse_face_row','node_parent_coarse_edge_a','node_parent_coarse_edge_b'):
        if not np.array_equal(out_arrays[k][:len(X)],source[k]): raise AssertionError(f'node source prefix changed: {k}')
    if not np.array_equal(out_arrays['parent_coarse_face_row'],source['parent_coarse_face_row'][result['parent_face_rows']]): raise AssertionError('coarse face provenance changed')
    if not np.array_equal(out_arrays['parent_source_face_marker'],result['p084_parent_face_markers']): raise AssertionError('P084 parent markers were not propagated exactly')
    for key in ('coarse_boundary_anchor_rows','coarse_boundary_anchor_source_nodes','typed_coarse_anchor_output_rows_018','typed_coarse_anchor_source_ids_018'):
        if key in source and not np.array_equal(out_arrays[key],source[key]): raise AssertionError(f'anchor lineage changed: {key}')
    # Persist the complete candidate before exact checks so failures retain the geometry and provenance.
    with out.open('xb') as f: np.savez_compressed(f,**out_arrays)
    audit=exact_self_audit(out_arrays['points_world_m'],out_arrays['source_O_faces'])
    pairs=np.asarray(audit.get('triangle_pairs',[]),dtype=np.int64).reshape(-1,2)
    report_out={
      'status':'REFINED_EXACT_SELF_CLEAR' if len(pairs)==0 and audit.get('complete_surface_audit') else 'REFINED_SELF_AUDIT_FAILED',
      'candidate_input':str(cand),'candidate_sha256':csha,'face_report_input':str(fr),'face_report_sha256':fsha,
      'crossing_bank_pair_count':pair_count,'unique_crossed_face_count':int(len(crossed)),
      'affected_parent_face_count':int(len(result['affected_parent_face_rows'])),'selected_edge_count':int(len(result['selected_edges'])),
      'added_edge_midpoints':int(len(result['selected_edges'])),'added_face_centroids':int(len(result['affected_parent_face_rows'])),
      'old_nodes':int(len(X)),'new_nodes':int(len(out_arrays['points_world_m'])),'old_faces':int(len(F)),'new_faces':int(len(out_arrays['source_O_faces'])),
      'boundary_components_before':int(result['boundary_components_before']),'boundary_components_after':int(result['boundary_components_after']),
      'boundary_edges_before':int(result['boundary_edges_before']),'boundary_edges_after':int(result['boundary_edges_after']),
      'max_parent_area_relative_drift':float(np.max(result['relative_parent_area_error'])),
      'min_child_area_m2':float(np.min(0.5*np.linalg.norm(np.cross(out_arrays['points_world_m'][out_arrays['source_O_faces'][:,1]].astype(np.float64)-out_arrays['points_world_m'][out_arrays['source_O_faces'][:,0]],out_arrays['points_world_m'][out_arrays['source_O_faces'][:,2]].astype(np.float64)-out_arrays['points_world_m'][out_arrays['source_O_faces'][:,0]]),axis=1))),
      'exact_self_pair_count':int(len(pairs)),'exact_self_audit_complete':bool(audit.get('complete_surface_audit')),
      'self_predicate_sha256':sha256(PREDICATE),'p084_owner_sha256':sha256(P084_OWNER),'candidate_output_sha256':sha256(out),'candidate_output':str(out),
      'surface_geometry_changed_only_by_f32_affine_subdivision':True,'existing_vertices_bitwise_unchanged':True,
      'source_prolongation_rows':int(len(out_arrays['points_world_m'])),'source_prolongation_columns':int(len(X)),
      'physical_steps':0,'geometry_admitted':False,'elapsed_seconds':time.monotonic()-started,
      'limitations':['Subdivision preserves the represented coarse surface within Float32 midpoint/centroid rounding; it does not remove bone intersections by itself.','No native runtime, material, mass, volume, or tissue-physics admission was performed.']
    }
    report_out['crossed_parent_face_rows']=crossed.tolist()
    report_out['affected_parent_face_rows']=result['affected_parent_face_rows'].tolist()
    report_out['self_witness_pairs']=pairs[:200].tolist()
    rp.write_text(json.dumps(report_out,indent=2)+'\n')
    np.save(out.parent/'exact-self-pairs.npy',pairs)
    print(json.dumps(report_out,indent=2))
    if report_out['status']!='REFINED_EXACT_SELF_CLEAR': raise SystemExit(2)

if __name__=='__main__': main()
