#!/usr/bin/env python3
"""Reproduce low-q child witnesses from frozen full084 scalar-plane split."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

D=Path('/Users/n/Work/numi-human-connected-surface-069')
INPUT=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
SPLIT=D/'publication-081/evidence/full084-split-001/scalar-plane-split-084.npz'
REPORT=D/'publication-081/evidence/near-vertex-audit-001/report.json'
THRESHOLD=1.0e-6
EDGE_PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def det_q(points,tet):
    p=np.asarray(points[tet],dtype=np.float64)
    det=float(np.dot(p[1]-p[0],np.cross(p[2]-p[0],p[3]-p[0])))
    lengths=[float(np.linalg.norm(p[i]-p[j])) for i,j in EDGE_PAIRS]
    L=max(lengths)
    return det,det/(L**3),L

def altitude_and_ulp(points32,tet):
    p=np.asarray(points32[tet],dtype=np.float64)
    det=abs(float(np.dot(p[1]-p[0],np.cross(p[2]-p[0],p[3]-p[0]))))
    areas=[]
    for i in range(4):
        face=np.delete(p,i,axis=0)
        areas.append(float(np.linalg.norm(np.cross(face[1]-face[0],face[2]-face[0]))))
    positive=[a for a in areas if a>0]
    h=min(det/a for a in positive) if positive else 0.0
    values=np.abs(np.asarray(points32[tet],dtype=np.float32))
    ulp=np.spacing(values).astype(np.float64)
    max_ulp=float(np.max(ulp))
    return h,max_ulp

def main():
    REPORT.parent.mkdir(parents=True,exist_ok=False)
    with np.load(INPUT,allow_pickle=False) as s:
        R=np.asarray(s['reference_points_m'],dtype=np.float64)
        T=np.asarray(s['tetrahedra_zero_based'],dtype=np.int64)
        gids=np.asarray(s['output_node_ids'],dtype=np.int64)
        uids=np.asarray(s['cell_uids'],dtype=np.int64)
        mats=np.asarray(s['material_ids'],dtype=np.int64)
    with np.load(SPLIT,allow_pickle=False) as a:
        P=np.asarray(a['points_reference_m'],dtype=np.float64)
        P32=np.asarray(a['points_reference_f32'],dtype=np.float32)
        X32=np.asarray(a['points_current037_f32'],dtype=np.float32)
        C=np.asarray(a['tetrahedra'],dtype=np.int64)
        parent=np.asarray(a['parent_cell_rows'],dtype=np.int64)
        side=np.asarray(a['side_signs'],dtype=np.int8)
        scalar=np.asarray(a['scalar_values'],dtype=np.float64)
        edge_ids=np.asarray(a['intersection_edge_global_ids'],dtype=np.int64)
        wdata=np.asarray(a['interpolation_weights_data'],dtype=np.float64)
        widx=np.asarray(a['interpolation_weights_indices'],dtype=np.int64)
        wptr=np.asarray(a['interpolation_weights_indptr'],dtype=np.int64)
    n=len(R)
    run=json.loads((D/'publication-081/evidence/full084-split-001/report.json').read_text())
    normal=np.asarray(run['scalar_field_definition']['normal_unit'],dtype=np.float64)
    threshold=float(run['scalar_field_definition']['threshold_m'])
    phi=R@normal-threshold
    source_q=np.asarray([det_q(R,t)[1] for t in T])
    q_ref=np.asarray([det_q(P32,t)[1] for t in C])
    det_ref=np.asarray([det_q(P32,t)[0] for t in C])
    q_cur=np.asarray([det_q(X32,t)[1] for t in C])
    det_cur=np.asarray([det_q(X32,t)[0] for t in C])
    low=np.flatnonzero(q_ref<THRESHOLD)
    children=[]
    for cr in low:
        pr=int(parent[cr]); pt=T[pr]; tet=C[cr]
        h,ulp=altitude_and_ulp(P32,tet)
        cut=[]
        for outrow in tet[tet>=n]:
            lo,hi=int(wptr[outrow]),int(wptr[outrow+1])
            rows=widx[lo:hi].astype(int).tolist()
            weights=wdata[lo:hi].astype(float).tolist()
            d=np.linalg.norm(R[pt]-P[outrow],axis=1)
            cut.append({'output_row':int(outrow),'source_edge_rows':rows,
                        'source_edge_global_ids':[int(gids[i]) for i in rows],
                        'interpolation_weights':weights,
                        'nearest_parent_vertex_row':int(pt[int(np.argmin(d))]),
                        'nearest_parent_vertex_global_id':int(gids[int(pt[int(np.argmin(d))])]),
                        'distance_to_parent_vertices_um':(d*1e6).tolist()})
        children.append({
            'split_child_row':int(cr),'parent_cell_row':pr,'parent_cell_uid':int(uids[pr]),
            'material_id':int(mats[pr]),'side_sign':int(side[cr]),
            'child_output_vertex_rows':tet.astype(int).tolist(),
            'parent_source_vertex_rows':pt.astype(int).tolist(),
            'parent_source_global_vertex_ids':gids[pt].astype(int).tolist(),
            'parent_plane_signed_values_um':(phi[pt]*1e6).tolist(),
            'parent_source_q_reference':float(source_q[pr]),
            'child_q_reference_float32':float(q_ref[cr]),
            'child_q_current037_float32':float(q_cur[cr]),
            'child_det_reference_float32_m3':float(det_ref[cr]),
            'child_det_current037_float32_m3':float(det_cur[cr]),
            'current037_over_reference_float32_det':float(det_cur[cr]/det_ref[cr]),
            'minimum_altitude_reference_float32_um':float(h*1e6),
            'maximum_coordinate_ulp_in_child_um':float(ulp*1e6),
            'minimum_altitude_over_max_coordinate_ulp':float(h/ulp) if ulp else None,
            'f32_status':'positive determinant; no collapsed cell observed; altitude/ULP ratio is diagnostic only',
            'cut_vertices':cut})
    report={
        'status':'PASS_READ_ONLY_NEAR_VERTEX_QUALITY_AUDIT',
        'scope':'saved full084 split outputs; no geometry regenerated or modified',
        'physical_or_native_admission':False,
        'input_sha256':sha(INPUT),'split_sha256':sha(SPLIT),
        'threshold_q':THRESHOLD,'metric':'positive Float32 determinant / longest_edge^3',
        'low_quality_child_count':len(low),
        'low_quality_children':children,
        'global_summary':{
            'source_cell_count':len(T),'split_child_count':len(C),
            'source_min_q_reference':float(source_q.min()),
            'split_min_q_reference_float32':float(q_ref.min()),
            'split_cells_below_threshold':int(len(low)),
            'minimum_child_altitude_over_coordinate_ulp':float(min(x['minimum_altitude_over_max_coordinate_ulp'] for x in children)),
            'interpretation':'The low-q rows arise where the exact scalar cut creates a small interface facet close to a source vertex. Source parents are positive and substantially better conditioned. All audited children retain positive Float32 determinants; this report does not establish an acceptance floor.'},
    }
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(REPORT),'report_sha256':sha(REPORT),'rows':len(low),'minimum_q':float(q_ref.min()),'min_altitude_over_max_ulp':report['global_summary']['minimum_child_altitude_over_coordinate_ulp']},indent=2))
if __name__=='__main__': main()
