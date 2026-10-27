#!/usr/bin/env python3
"""Apply exact 545 all-185 exit targets to newly appended 585 edge midpoints only.

This emits an unadmitted point-coordinate proposal. It preserves the complete
refined NPZ payload except points_world_m and requires subsequent full checks.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
SELECTOR=ROOT/'D/surface-clearance-546/select_batch_endpoint_operations546.py'
SELECTOR_SHA='9c3257524e95b3e00ee41c154caa90cfbc4fcd1031719d46a591338232e03f8b'
TARGET_ADAPTER=ROOT/'D/surface-clearance-546/direct-depth-targets-546/run_direct_depth_targets546.py'
TARGET_ADAPTER_SHA='0708d259c7bc7dd6341b8cac737f6516ac611011e6d4f599e05530cf91106ac5'
TARGET_BUILDER=ROOT/'D/contained-clearance-545/build_direct_depth_targets545.py'
TARGET_BUILDER_SHA='2864d9547d2715c61c9c27b17bcfd11fac250197c47f4fd4870f8f6898cecb60'
MULTISOURCE=ROOT/'D/source-outward-proposal-500/build_outward_targets_multi_source_546.py'
MULTISOURCE_SHA='dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68'
OWNER500=ROOT/'D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py'
OWNER500_SHA='454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf'
REFINER585=ROOT/'D/surface-clearance-546/edge-only-refinement-585/edge_only_refinement585.py'
REFINER585_SHA='07d466ba25c7d565bb8be61c9ff87a7725699f13e4080587a48cbbeec24056a3'
ADAPTER546_SHA='06081873eaf0d5452fbf4beee3d580a0eba97c3a546a9f7e7a06a0c9f548cbc6'
P037_SHA='d3a6e0c31e34f2165d853ffaf411ad67faaa1d307a3af1a6c79ba41a129f4f31'
P084_SHA='eab8deda28164964051a6fa6ff346b8d94bc8bede04e896411dfdd02d46594f6'
ANCHORS=('coarse_boundary_anchor_rows','coarse_boundary_anchor_source_nodes',
         'typed_coarse_anchor_output_rows_018','typed_coarse_anchor_source_ids_018')
PARENT_Q_PREFIX='prior_input_parent__'
BASE_586=ROOT/'D/surface-clearance-546/midpoint-exit-relocation-586/relocate_midpoint_targets586.py'
BASE_586_SHA='b1c888916075922366eefbda9fe3c4d2b51ab044bcd77e5ee222f6f5a4bdb18b'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def raw_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha_i64(a): return hashlib.sha256(np.ascontiguousarray(a,dtype='<i8').tobytes()).hexdigest()
def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def readj(p):
    x=json.loads(Path(p).read_text())
    if not isinstance(x,dict): raise ValueError(f'expected JSON object: {p}')
    return x
def loadnpz(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def load_selector():
    if sha(SELECTOR)!=SELECTOR_SHA: raise ValueError('pinned 546 target validator changed')
    s=importlib.util.spec_from_file_location('_selector546_for586',SELECTOR)
    if s is None or s.loader is None: raise ImportError(str(SELECTOR))
    m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def validate_owner_pins():
    if sha(BASE_586)!=BASE_586_SHA: raise ValueError('frozen 586 v1 helper changed')
    for p,h,n in ((TARGET_ADAPTER,TARGET_ADAPTER_SHA,'546 direct-depth adapter'),
                  (TARGET_BUILDER,TARGET_BUILDER_SHA,'545 direct-depth builder'),
                  (MULTISOURCE,MULTISOURCE_SHA,'546 multi-source extension'),
                  (OWNER500,OWNER500_SHA,'500 source normal owner'),
                  (REFINER585,REFINER585_SHA,'585 edge-only refiner')):
        if sha(p)!=h: raise ValueError(f'pinned {n} changed')

def validate_current_source_prolongation(parent, candidate, parent_points, midpoint_rows, parent_edges, midpoint_weights):
    """Check current Q rows equal edge-midpoint P composed with the parent's Q."""
    keys = tuple("source_prolongation_" + s for s in ("indptr", "node_rows", "weights"))
    if any(k not in parent or k not in candidate for k in keys):
        raise ValueError("current/parent source prolongation Q arrays missing")
    pp, pi, pw = (np.asarray(parent[k]) for k in keys)
    cp, ci, cw = (np.asarray(candidate[k]) for k in keys)
    n = int(parent_points)
    ncurrent = len(np.asarray(candidate.get("points_world_m")))
    def valid(ptr, ids, weights, point_rows, label):
        if ptr.dtype.kind not in "iu" or ids.dtype.kind not in "iu":
            raise ValueError(f"{label} Q pointers/IDs must be integer")
        if ptr.shape != (point_rows + 1,) or int(ptr[0]) != 0 or int(ptr[-1]) != len(ids) or np.any(ptr[1:] < ptr[:-1]):
            raise ValueError(f"{label} Q CSR pointers malformed")
        if weights.ndim != 1 or weights.shape != ids.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"{label} Q weights malformed")
        if np.any(ids < 0):
            raise ValueError(f"{label} Q source IDs negative")
        for row in range(point_rows):
            a,b=int(ptr[row]),int(ptr[row+1])
            if b <= a or len(np.unique(ids[a:b])) != b-a or np.any(ids[a+1:b] <= ids[a:b-1]):
                raise ValueError(f"{label} Q row must be nonempty, sorted, unique CSR")
            if abs(float(weights[a:b].sum())-1.0) > 8*np.finfo(np.float64).eps:
                raise ValueError(f"{label} Q row does not sum to one")
    valid(pp,pi,pw,n,"parent")
    valid(cp,ci,cw,ncurrent,"current")
    if cp[:n+1].dtype != pp.dtype or not same(cp[:n+1],pp):
        raise ValueError("current Q prefix row pointers differ from parent")
    prefix_nnz=int(pp[n])
    if not same(ci[:prefix_nnz],pi) or not same(cw[:prefix_nnz],pw):
        raise ValueError("current Q prefix payload differs from parent")
    rows=np.asarray(midpoint_rows,dtype=np.int64)
    edges=np.asarray(parent_edges,dtype=np.int64)
    edgew=np.asarray(midpoint_weights,dtype=np.float64)
    if edges.shape != (len(rows),2) or edgew.shape != (len(rows),2):
        raise ValueError("current Q midpoint parent arrays have wrong shape")
    max_error=0.0
    for j,row in enumerate(rows):
        expected={}
        for endpoint,scale in zip(edges[j],edgew[j]):
            if endpoint < 0 or endpoint >= n or scale < 0 or not np.isfinite(scale):
                raise ValueError("current Q midpoint parent/weight invalid")
            a,b=int(pp[endpoint]),int(pp[endpoint+1])
            for col,value in zip(pi[a:b],pw[a:b]):
                expected[int(col)]=expected.get(int(col),0.0)+float(scale)*float(value)
        expected={k:v for k,v in expected.items() if v>0.0}
        expected_ids=np.asarray(sorted(expected),dtype=ci.dtype)
        expected_weights=np.asarray([expected[int(k)] for k in expected_ids],dtype=np.float64)
        a,b=int(cp[row]),int(cp[row+1])
        got_ids=ci[a:b]
        got_weights=cw[a:b].astype(np.float64,copy=False)
        if not np.array_equal(got_ids,expected_ids):
            raise ValueError(f"current Q midpoint row {int(row)} support differs from edge-parent Q composition")
        err=float(np.max(np.abs(got_weights-expected_weights),initial=0.0))
        max_error=max(max_error,err)
        if err > 8*np.finfo(np.float64).eps:
            raise ValueError(f"current Q midpoint row {int(row)} weights differ from edge-parent Q composition")
    return {"status":"CURRENT_Q_EQUALS_EDGE_MIDPOINT_COMPOSITION",
            "parent_prefix_rows":n,"midpoint_rows":rows.tolist(),
            "source_prolongation_nnz":int(cp[-1]),"maximum_weight_abs_error":max_error,
            "rounding_tolerance":float(8*np.finfo(np.float64).eps)}

def validate_refinement(parent_path,candidate_path,report_path,parent,candidate,r):
    ph,ch=sha(parent_path),sha(candidate_path)
    if r.get('status')!='REFINEMENT_CONSTRUCTION_READY' or r.get('physical_steps')!=0 or r.get('geometry_moved') is not False:
        raise ValueError('585 report is not a geometry-unchanged construction receipt')
    if r.get('input_candidate_sha256')!=ph or r.get('output_candidate_sha256')!=ch:
        raise ValueError('585 parent/output candidate hash mismatch')
    if Path(r.get('input_candidate_path','')).resolve()!=Path(parent_path).resolve() or Path(r.get('output_candidate_path','')).resolve()!=Path(candidate_path).resolve():
        raise ValueError('585 parent/output candidate path mismatch')
    for k,h in (('edge_only_refinement585_sha256',REFINER585_SHA),('pinned_546_adapter_sha256',ADAPTER546_SHA),('pinned_037_refiner_sha256',P037_SHA),('pinned_084_owner_sha256',P084_SHA)):
        if r.get(k)!=h: raise ValueError(f'585 source pin mismatch: {k}')
    if r.get('added_nodes_are_only_edge_midpoints') is not True or r.get('added_face_centroids')!=0 or r.get('old_point_prefix_bitwise_unchanged') is not True or r.get('existing_node_coordinates_moved') is not False:
        raise ValueError('585 receipt does not establish edge-only additions with fixed old prefix')
    pX=np.asarray(parent.get('points_world_m')); X=np.asarray(candidate.get('points_world_m'))
    if pX.dtype!=np.float32 or X.dtype!=np.float32 or pX.ndim!=2 or X.ndim!=2 or pX.shape[1:]!=(3,) or X.shape[1:]!=(3,) or not np.isfinite(pX).all() or not np.isfinite(X).all():
        raise ValueError('parent/refined point arrays must be finite Float32 Nx3')
    n=len(pX)
    if (r.get('input_points')!=n or r.get('parent_prefix_node_count')!=n or r.get('old_node_count')!=n or r.get('new_node_count')!=len(X)):
        raise ValueError('585 point counts do not match candidate domains')
    if len(X)<=n or not same(pX,X[:n]) or r.get('parent_prefix_points_bytes_sha256')!=raw_sha(pX):
        raise ValueError('585 old point prefix does not match parent bitwise')
    rows0=r.get('edge_midpoint_rows')
    if not isinstance(rows0,list) or any(type(x) is not int for x in rows0): raise ValueError('585 midpoint rows are not an integer list')
    rows=np.asarray(rows0,dtype=np.int64); expected=np.arange(n,len(X),dtype=np.int64)
    if len(rows)!=r.get('edge_midpoint_rows_count') or not np.array_equal(rows,expected) or r.get('edge_midpoint_rows_sha256')!=sha_i64(rows):
        raise ValueError('585 midpoint rows do not uniquely cover appended suffix')
    if r.get('selected_edge_count')!=len(rows): raise ValueError('585 edge and midpoint counts differ')
    E=np.asarray(candidate.get('refinement_node_parent_edge_rows')); W=np.asarray(candidate.get('refinement_node_parent_edge_weights')); K=np.asarray(candidate.get('refinement_node_kind'))
    if E.shape!=(len(X),2) or E.dtype.kind not in 'iu' or W.shape!=(len(X),2) or not np.isfinite(W).all() or K.shape!=(len(X),) or K.dtype.kind not in 'iu':
        raise ValueError('candidate lacks current midpoint lineage fields')
    E=E[rows].astype(np.int64,copy=False)
    if np.any(E<0) or np.any(E>=n) or np.any(E[:,0]==E[:,1]) or not np.all(W[rows]==.5) or not np.all(K[rows]==1):
        raise ValueError('appended rows are not 0.5/0.5 midpoint nodes over parent prefix')
    if r.get('edge_midpoint_parent_edges_sha256')!=sha_i64(E): raise ValueError('585 parent-edge rows hash mismatch')
    midpoint=((pX[E[:,0]].astype(np.float64)+pX[E[:,1]].astype(np.float64))*.5).astype(np.float32)
    if not same(X[rows],midpoint): raise ValueError('appended points do not replay from midpoint parents')
    # 546 archives the prior Q/P under prior_input_parent__, then exposes new Q.
    for suffix in ('indptr','node_rows','weights','semantics'):
        k='source_prolongation_'+suffix; archived=PARENT_Q_PREFIX+k
        if k in parent and (archived not in candidate or not same(parent[k],candidate[archived])):
            raise ValueError(f'prior Q/P archive mismatch: {suffix}')
    current_q_report=validate_current_source_prolongation(parent,candidate,n,rows,E,W[rows])
    ref='area_reference_037_points_world_m_f64'; ref_faces='area_reference_037_source_O_faces'
    if ref not in parent or ref not in candidate or ref_faces not in candidate: raise ValueError('037 reference arrays missing')
    rp=np.asarray(parent[ref]); rc=np.asarray(candidate[ref])
    if rp.dtype!=np.float64 or rc.dtype!=np.float64 or rp.shape!=(n,3) or rc.shape!=(len(X),3) or not same(rp,rc[:n]):
        raise ValueError('037 reference prefix/domain mismatch')
    expected_ref=rp[E[:,0]]*W[rows,0,None]+rp[E[:,1]]*W[rows,1,None]
    if not same(rc[rows],expected_ref): raise ValueError('037 new reference points do not follow midpoint Q')
    if r.get('area_reference_uses_same_Q') is not True or r.get('area_reference_faces_match_candidate_exactly') is not True or not np.array_equal(candidate[ref_faces],candidate['source_O_faces']):
        raise ValueError('585 report does not certify current 037 face/Q correspondence')
    for k in ('area_reference_037_semantics','area_reference_037_input_sha256'):
        if k in parent and (k not in candidate or not same(parent[k],candidate[k])): raise ValueError(f'037 source-reference pin changed: {k}')
    for k in ANCHORS:
        if k in parent and (k not in candidate or not same(parent[k],candidate[k])): raise ValueError(f'anchor field changed in 585: {k}')
    return rows

def select_midpoint_targets(mid_rows,target_rows,target_points,anchors,npoints):
    m=np.asarray(mid_rows); r=np.asarray(target_rows); T=np.asarray(target_points); a=np.asarray(anchors)
    if m.ndim!=1 or m.dtype.kind not in 'iu' or len(np.unique(m))!=len(m): raise ValueError('midpoint rows must be unique integers')
    if r.ndim!=1 or r.dtype.kind not in 'iu' or len(np.unique(r))!=len(r): raise ValueError('target rows must be unique integers')
    if T.dtype!=np.float32 or T.shape!=(len(r),3) or not np.isfinite(T).all(): raise ValueError('target endpoints must be finite Float32 row-aligned points')
    if a.ndim!=1 or a.dtype.kind not in 'iu' or len(np.unique(a))!=len(a): raise ValueError('typed anchors must be unique integer rows')
    if np.any(m<0) or np.any(m>=npoints) or np.any(r<0) or np.any(r>=npoints) or np.any(a<0) or np.any(a>=npoints): raise ValueError('target/midpoint/anchor outside point domain')
    selected=np.intersect1d(m.astype(np.int64),r.astype(np.int64),assume_unique=True)
    if not len(selected): raise ValueError('no appended midpoint row has a certified outside endpoint')
    if np.intersect1d(selected,a.astype(np.int64)).size: raise ValueError('certified endpoint targets a typed anchor')
    lookup={int(row):T[i].copy() for i,row in enumerate(r)}
    return selected,np.asarray([lookup[int(row)] for row in selected],dtype=np.float32)

def relocate_points(points,rows,targets):
    X=np.asarray(points); rows=np.asarray(rows); T=np.asarray(targets)
    if X.dtype!=np.float32 or X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all(): raise ValueError('candidate points must be finite Float32 Nx3')
    if rows.ndim!=1 or rows.dtype.kind not in 'iu' or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(X)): raise ValueError('selected point rows invalid')
    if T.dtype!=np.float32 or T.shape!=(len(rows),3) or not np.isfinite(T).all(): raise ValueError('selected endpoints invalid')
    Y=X.copy(); Y[rows.astype(np.int64)]=T; keep=np.ones(len(X),bool); keep[rows.astype(np.int64)]=False
    if not same(Y[keep],X[keep]): raise AssertionError('non-target coordinate changed')
    return Y

def source_alignment_diagnostic(start_world_f32_m, target_world_f32_m,
                                 source_normal_world_unit, proposal_direction_world_unit):
    start=np.asarray(start_world_f32_m,dtype=np.float64)
    target=np.asarray(target_world_f32_m,dtype=np.float64)
    normal=np.asarray(source_normal_world_unit,dtype=np.float64)
    proposal=np.asarray(proposal_direction_world_unit,dtype=np.float64)
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (start,target,normal,proposal)):
        raise ValueError('source-normal alignment inputs must be finite 3-vectors')
    nn=float(np.linalg.norm(normal)); pn=float(np.linalg.norm(proposal))
    delta=target-start; dn=float(np.linalg.norm(delta))
    if nn<=0 or pn<=0 or dn<=0: raise ValueError('source-normal alignment has a zero vector')
    n=normal/nn; p=proposal/pn; d=delta/dn
    source_proposal=float(np.dot(n,p)); source_applied=float(np.dot(n,d))
    return {'source_normal_dot_proposal_direction':source_proposal,
            'source_normal_dot_applied_direction':source_applied,
            'source_normal_dot_applied_displacement_m':float(np.dot(n,delta)),
            'applied_displacement_m':dn,
            'proposal_applied_cosine':float(np.dot(p,d)),
            'nonpositive_source_proposal_alignment':source_proposal<=0.0,
            'nonpositive_source_applied_alignment':source_applied<=0.0,
            'source_hemisphere_requirement_applied':False,
            'diagnostic_only':True}

def validate_targets_against_state(state,candidate,target_report_path,targets_path,selector):
    st,summary,surface=selector.validate_baseline(state)
    cpath=Path(st['candidate']); ch=st['candidate_sha256']; target_report_path=Path(target_report_path); targets_path=Path(targets_path)
    tr,tz,rows,T=selector.validate_targets(target_report_path,targets_path,ch)
    inputs=tr.get('inputs',{}); summary_path=Path(st['point_summary']); qdir=summary_path.parent/'query'
    files={'point_summary_sha256':summary_path,'query_receipt_sha256':qdir/'query-receipt.json','query_points_sha256':qdir/'points.csv','query_nodes_sha256':qdir/'nodes.csv','query_pairs_sha256':qdir/'pairs.csv'}
    for k,p in files.items():
        if inputs.get(k)!=sha(p): raise ValueError(f'545 target input pin mismatch: {k}')
    ext=tr.get('multi_parent_source_normal_extension')
    if not isinstance(ext,dict): raise ValueError('point targets lack 546 multi-parent source provenance')
    for k,h in (('pinned_545_builder_sha256',TARGET_BUILDER_SHA),('pinned_546_extension_sha256',MULTISOURCE_SHA),('pinned_500_owner_sha256',OWNER500_SHA)):
        if ext.get(k)!=h: raise ValueError(f'point target source pin mismatch: {k}')
    if tr.get('direction_mode')!='direct-depth' or tr.get('physical_steps')!=0: raise ValueError('point target report is not direct-depth proposal-only')
    if ext.get('direct_depth_direction_unchanged') is not True or ext.get('exact_185_bank_ray_and_endpoint_owner_unchanged') is not True: raise ValueError('point-target all185/direction contract missing')
    X=np.asarray(candidate['points_world_m']); F=np.asarray(candidate['source_O_faces']);
    if summary.get('candidate_sha256')!=ch or summary.get('complete_surface_audit') is not True or summary.get('exact_PLC_surface_pairs')!=0: raise ValueError('baseline 130 is stale, incomplete, or self-intersecting')
    if summary.get('points')!=len(X) or summary.get('faces')!=len(F): raise ValueError('130 summary domain mismatch')
    receipt=readj(qdir/'query-receipt.json')
    if receipt.get('identity_domain')!='mechanics_node_row' or receipt.get('query_rows')!=len(X): raise ValueError('130 query receipt domain/count mismatch')
    for key,file in (('points_sha256',qdir/'points.csv'),('nodes_sha256',qdir/'nodes.csv'),('pairs_sha256',qdir/'pairs.csv')):
        if receipt.get(key)!=sha(file): raise ValueError(f'130 query CSV receipt mismatch: {key}')
    with (qdir/'points.csv').open(newline='') as f: point_records=list(csv.DictReader(f))
    with (qdir/'nodes.csv').open(newline='') as f: node_records=list(csv.DictReader(f))
    with (qdir/'pairs.csv').open(newline='') as f: pair_records=list(csv.DictReader(f))
    if len(point_records)!=len(X) or len(node_records)!=len(X): raise ValueError('130 point/node CSV domains incomplete')
    bits=np.ascontiguousarray(X).view(np.uint32)
    inside=set()
    for i,(p,n) in enumerate(zip(point_records,node_records)):
        if int(p['query_row'])!=i or int(n['query_row'])!=i: raise ValueError('130 row identity mismatch')
        if any(int(p[k])!=int(bits[i,j]) for j,k in enumerate(('x_bits_u32','y_bits_u32','z_bits_u32'))): raise ValueError(f'130 point bits mismatch at {i}')
        if int(n['total_bone_banks'])!=185 or int(n['ambiguous_banks']) or int(n['invalid_banks']): raise ValueError(f'130 node row not complete all185 at {i}')
        total=sum(int(n[k]) for k in ('aabb_proven_outside_banks','parity_outside_banks','inside_banks','ambiguous_banks','invalid_banks'))
        if total!=185: raise ValueError(f'130 node bank accounting mismatch at {i}')
    for p in pair_records:
        row=int(p['query_row'])
        if row<0 or row>=len(X) or int(p.get('ambiguous','0'))!=0 or p['classification'] not in ('inside','outside'): raise ValueError('invalid 130 pair classification')
        if p['classification']=='inside': inside.add(row)
    if len(np.unique(rows))!=len(rows) or set(map(int,rows))!=inside: raise ValueError('545 target rows do not equal complete 130 inside-node set')
    return tr,rows,T,np.asarray(sorted(inside),dtype=np.int64),tz

def run(state_path,parent_path,candidate_path,refinement_path,target_report_path,targets_path,out):
    start=time.monotonic()
    if out.exists(): raise FileExistsError(out)
    validate_owner_pins(); selector=load_selector(); st,_,_=selector.validate_baseline(state_path)
    if Path(st['candidate']).resolve()!=candidate_path.resolve(): raise ValueError('state candidate path mismatch')
    parent=loadnpz(parent_path); candidate=loadnpz(candidate_path); rr=readj(refinement_path)
    mids=validate_refinement(parent_path,candidate_path,refinement_path,parent,candidate,rr)
    tr,targets_rows,targets,inside,target_archive=validate_targets_against_state(state_path,candidate,target_report_path,targets_path,selector)
    anchors=np.asarray(candidate.get('typed_coarse_anchor_output_rows_018'))
    selected,T=select_midpoint_targets(mids,targets_rows,targets,anchors,len(candidate['points_world_m']))
    if not np.array_equal(selected,np.intersect1d(mids,inside,assume_unique=True)): raise AssertionError('selected new inside rows differ from audit')
    X=np.asarray(candidate['points_world_m']); Y=relocate_points(X,selected,T)
    target_index={int(row):i for i,row in enumerate(targets_rows)}
    target_records={int(item['query_row']):item for item in tr.get('targets',[])}
    normal_array=np.asarray(target_archive['source_surface_normal_world_unit'])
    proposal_array=np.asarray(target_archive['proposal_direction_world_unit'])
    source_mask=np.asarray(target_archive['source_normal_available_mask'])
    if normal_array.shape!=(len(targets_rows),3) or proposal_array.shape!=(len(targets_rows),3) or source_mask.shape!=(len(targets_rows),):
        raise ValueError('545 target arrays lack row-aligned source/proposal direction fields')
    alignment_rows=[]
    for row in selected:
        i=target_index[int(row)]; item=target_records[int(row)]
        start=np.asarray(item.get('start_world_f32_m'),dtype=np.float32)
        if start.shape!=(3,) or not same(start,X[int(row)]): raise ValueError(f'545 target start differs from candidate row {int(row)}')
        target_item=np.asarray(item.get('target_world_f32_m'),dtype=np.float32)
        if not same(target_item,T[i]): raise ValueError(f'545 target report endpoint differs at row {int(row)}')
        if not bool(source_mask[i]): raise ValueError(f'545 lacks registered source normal for midpoint row {int(row)}')
        normal=np.asarray(item.get('source_surface_normal_world_unit'),dtype=np.float64)
        proposal=np.asarray(item.get('proposal_direction_world_unit'),dtype=np.float64)
        if not np.array_equal(normal,np.asarray(normal_array[i],dtype=np.float64)) or not np.array_equal(proposal,np.asarray(proposal_array[i],dtype=np.float64)):
            raise ValueError(f'545 direction report/array differs at row {int(row)}')
        diag=source_alignment_diagnostic(start,T[i],normal,proposal)
        alignment_rows.append({'query_row':int(row),**diag,
            'source_surface_normal_world_unit':normal.tolist(),
            'proposal_direction_world_unit':proposal.tolist(),
            'start_world_f32_m':start.tolist(),'target_world_f32_m':T[i].tolist()})
    changed=np.flatnonzero(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1))
    expected_changed=selected[np.any(T.view(np.uint32)!=X[selected].view(np.uint32),axis=1)]
    if not np.array_equal(changed,expected_changed): raise AssertionError('only selected midpoint coordinates may change')
    payload={k:v.copy() for k,v in candidate.items()};payload['points_world_m']=Y
    out.mkdir(parents=True,exist_ok=False); candidate_out=out/'candidate.npz'
    with candidate_out.open('xb') as f: np.savez_compressed(f,**payload)
    emitted=loadnpz(candidate_out)
    if set(emitted)!=set(candidate): raise AssertionError('relocated NPZ schema changed')
    for k in candidate:
        if k=='points_world_m':
            if not same(emitted[k],Y): raise AssertionError('emitted point array differs')
        elif not same(emitted[k],candidate[k]): raise AssertionError(f'nonpoint array changed: {k}')
    report={'status':'NEW_EDGE_MIDPOINT_EXIT_RELOCATION_PROPOSAL_UNADMITTED',
      'classification':'per-node construction proposal; relocated endpoints have pinned 545 all185 certificates, while the complete output mesh still requires full checks',
      'candidate_path':str(candidate_out.resolve()),'candidate_sha256':sha(candidate_out),
      'baseline_state':str(state_path.resolve()),'baseline_state_sha256':sha(state_path),
      'input_refined_candidate':str(candidate_path.resolve()),'input_refined_candidate_sha256':sha(candidate_path),
      'parent_candidate':str(parent_path.resolve()),'parent_candidate_sha256':sha(parent_path),
      'refinement_report':str(refinement_path.resolve()),'refinement_report_sha256':sha(refinement_path),
      'target_report':str(target_report_path.resolve()),'target_report_sha256':sha(target_report_path),
      'targets_npz':str(targets_path.resolve()),'targets_sha256':sha(targets_path),
      'midpoint_rows':mids.tolist(),'midpoint_rows_sha256':sha_i64(mids),'current_source_prolongation_validation':current_q_report,'new_midpoint_inside_rows':selected.tolist(),
      'new_midpoint_inside_count':len(selected),'moved_rows':changed.astype(int).tolist(),'moved_row_count':len(changed),
      'preexisting_inside_targets_ignored':int(len(inside)-len(selected)),
      'each_moved_endpoint_certified_outside_all185':True,
      'source_normal_alignment_diagnostics':alignment_rows,
      'nonpositive_source_proposal_alignment_count':sum(bool(x['nonpositive_source_proposal_alignment']) for x in alignment_rows),
      'nonpositive_source_applied_alignment_count':sum(bool(x['nonpositive_source_applied_alignment']) for x in alignment_rows),
      'source_outward_hemisphere_was_not_a_545_direct_depth_gate':True,
      'old_point_prefix_bitwise_unchanged':True,
      'non_target_coordinates_bitwise_unchanged':True,'nonpoint_arrays_bitwise_unchanged':True,
      'parent_QP_archived_and_current_QP_preserved':True,'area_reference_037_Q_and_faces_validated':True,
      'typed_anchors_preserved_and_excluded':True,'native_admission':False,'geometry_admitted':False,'accepted_physical_steps':0,
      'required_next':'Run existing full 545 arithmetic/source/area/edge checks and complete exact 130/159 audits on this emitted candidate.',
      'limitations':['This applies certified endpoints only to appended midpoint nodes classified inside in the baseline query.','It does not certify post-relocation global self-intersection, area, edge-gradient, source-cap, or trust limits.'],
      'wall_seconds':time.monotonic()-start}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','parent-candidate','candidate','refinement-report','target-report','targets','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv)
    r=run(a.state,a.parent_candidate,a.candidate,a.refinement_report,a.target_report,a.targets,a.out)
    print(json.dumps({k:r[k] for k in ('status','new_midpoint_inside_count','moved_row_count','candidate_path','candidate_sha256','wall_seconds')},sort_keys=True))
if __name__=='__main__': main()
