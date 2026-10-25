"""Select connected authoring freedom from actual boundary witnesses.

This is an offline reconstruction workset, not a force, joint lock or runtime
constraint. All inner tissue cells retain their equations and nonlinear gates.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse

def select_boundary_witness_workset(node_count, proposed_free_nodes, outer_faces,
        crossing_edges, inside_nodes, contact_matrix, *, patch_rings=2):
    if isinstance(node_count,(bool,np.bool_)) or not isinstance(node_count,(int,np.integer)) or node_count<=0:
        raise ValueError('positive integer node count required')
    n=int(node_count)
    if isinstance(patch_rings,(bool,np.bool_)) or not isinstance(patch_rings,(int,np.integer)) or not 0<=patch_rings<=8:
        raise ValueError('patch rings must be an integer in [0,8]')
    def ids(value,width,name):
        v=np.asarray(value)
        shape_ok=v.ndim==1 if width==1 else v.ndim==2 and v.shape[1]==width
        if not shape_ok or v.dtype.kind not in 'iu' or np.any(v<0) or np.any(v>=n):
            raise ValueError('invalid '+name+' node domain')
        return v.astype(np.int64,copy=False)
    F=ids(proposed_free_nodes,1,'free')
    faces=ids(outer_faces,3,'outer face')
    edges=ids(crossing_edges,2,'crossing edge')
    inside=ids(inside_nodes,1,'inside')
    if not len(F) or len(np.unique(F))!=len(F) or not len(faces):
        raise ValueError('nonempty unique free nodes and outer faces required')
    if np.any(np.diff(np.sort(faces,axis=1),axis=1)==0):
        raise ValueError('degenerate outer face index tuple')
    C=sparse.csr_matrix(contact_matrix,dtype=float)
    if C.shape[1]!=3*n or not C.shape[0] or not np.isfinite(C.data).all():
        raise ValueError('invalid contact matrix')
    C.eliminate_zeros()
    if np.any(np.diff(C.indptr)==0):raise ValueError('empty contact row')
    outer=np.unique(faces)
    boundary_edges=np.unique(np.sort(np.concatenate(
        [faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    edge_set={tuple(e) for e in boundary_edges.tolist()}
    crossed=[e for e in edges.tolist() if tuple(sorted(e)) in edge_set]
    seeds=set(np.intersect1d(inside,outer).tolist())
    for edge in crossed:seeds.update(edge)
    outer_mask=np.zeros(n,dtype=bool);outer_mask[outer]=True
    all_outer_rows=0
    for k in range(C.shape[0]):
        nodes=np.unique(C.indices[C.indptr[k]:C.indptr[k+1]]//3)
        if np.all(outer_mask[nodes]):
            seeds.update(nodes.tolist());all_outer_rows+=1
    patch=np.asarray(sorted(seeds),dtype=np.int64)
    for _ in range(patch_rings):
        touching=np.isin(faces,patch).any(axis=1)
        patch=np.unique(faces[touching])
    # Expand only on the actual outer boundary; all proposed inner/free nodes
    # stay free. The caller re-evaluates this from refreshed exact witnesses.
    chosen=np.union1d(np.setdiff1d(F,outer),np.intersect1d(F,patch))
    if not len(chosen):raise ValueError('witness workset contains no free nodes')
    cols=(3*chosen[:,None]+np.arange(3)).ravel()
    active=C[:,cols].tocsr();active.eliminate_zeros()
    unaddressed=np.flatnonzero(np.diff(active.indptr)==0)
    report={'proposed_free_nodes':int(len(F)),'selected_free_nodes':int(len(chosen)),
        'outer_boundary_nodes':int(len(outer)),'outer_patch_seed_nodes':int(len(seeds)),
        'outer_patch_nodes':int(len(patch)),'selected_free_outer_nodes':int(np.intersect1d(chosen,outer).size),
        'anchored_proposed_outer_nodes':int(np.setdiff1d(np.intersect1d(F,outer),chosen).size),
        'crossing_boundary_edge_witnesses':len(crossed),'all_outer_contact_rows':all_outer_rows,
        'unaddressed_contact_row_ids':unaddressed.tolist(),'patch_rings':int(patch_rings),
        'runtime_constraints_added':False,'geometry_admitted':False}
    if len(unaddressed):raise ValueError('witness workset leaves contact rows without any free node')
    return chosen,report


def close_selected_boundary_fans(outer_faces, selected_face_rows):
    """Grow pinched patch boundaries to complete source vertex fans.

    Face rows remain source identities. This changes only the selected domain,
    not source coordinates, connectivity, or physical state.
    """
    from collections import defaultdict
    T=np.asarray(outer_faces);raw=np.asarray(selected_face_rows)
    if T.ndim!=2 or T.shape[1]!=3 or T.dtype.kind not in "iu":
        raise ValueError("integer Mx3 outer faces required")
    if T.size and (T.min()<0 or np.any(np.diff(np.sort(T,axis=1),axis=1)==0)):
        raise ValueError("invalid outer face vertices")
    if raw.ndim!=1 or raw.dtype.kind not in "iu" or len(np.unique(raw))!=len(raw):
        raise ValueError("selected face rows must be unique integers")
    if raw.size and (raw.min()<0 or raw.max()>=len(T)):
        raise ValueError("selected face row out of range")
    incident=defaultdict(set);global_edges=defaultdict(list)
    for fi,face in enumerate(T):
        for v in face:incident[int(v)].add(fi)
        for a,b in zip(face,np.roll(face,-1)):
            global_edges[tuple(sorted((int(a),int(b))))].append((int(a),int(b)))
    if any(len(v)>2 or (len(v)==2 and v[0]==v[1]) for v in global_edges.values()):
        raise ValueError("outer surface has nonmanifold or inconsistent edges")
    selected=set(map(int,raw));records=[]
    while selected:
        counts=defaultdict(int)
        for face in T[sorted(selected)]:
            for a,b in zip(face,np.roll(face,-1)):
                counts[tuple(sorted((int(a),int(b))))]+=1
        degree=defaultdict(int)
        for (a,b),count in counts.items():
            if count==1:degree[a]+=1;degree[b]+=1
        bad=sorted(v for v,d in degree.items() if d!=2)
        if not bad:break
        added=set().union(*(incident[v] for v in bad))-selected
        if not added:
            raise ValueError("source vertex fan itself has a nonmanifold boundary")
        records.append(dict(boundary_vertices=bad,added_source_face_rows=sorted(added)))
        selected.update(added)
    return np.array(sorted(selected),dtype=np.int64),dict(
        initial_face_count=int(len(raw)),final_face_count=int(len(selected)),
        closure_rounds=records,source_coordinates_or_connectivity_changed=False,
        physical_or_native_admission=False)
