#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/Users/n/Work/numi-human-connected-surface-069')
RUN = ROOT/'full-connected-clearance-150/run-017-conforming-crossing-refinement'
BASE = ROOT/'full-connected-clearance-150/run-017-point-constrained-soft-continuation/candidate.npz'
FACE_REPORT = ROOT/'full-connected-clearance-150/run-017-point-constrained-soft-continuation/final-face-audit-159/report.json'
FROZEN = RUN/'run-003/refined-candidate.npz'
DEPTH_OUTPUT = RUN/'registered-depth-adapter-018-run-002/registered-depths-refined.npz'
OWNER = RUN/'canonical-increment-019/matter/tools/human_skin_fat_boundary_refinement.py'
ADAPTER = RUN/'registered_depth_adapter_018.py'
OUT = RUN/'canonical-increment-019/parity-replay-020'
PINS = {
    'base_candidate': '3858c079054c4a87829fc554757104d74cb818073b1c8414acfd2d117fb3dbdd',
    'face_report': 'cfb3803f35ee465004f5dc60f73279035493a28de8076101618bc0adc6f09325',
    'run003_refined': '7a3a6fb208ad7e6af96fecf4bdcbab5aee8c3b340958a66b39676ea45a8176ec',
    'registered_depths': 'b54206114ed9cc62fb031ee2b038e5edf20f07c2146e23e4d0884894a0837b47',
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def compare(name, actual, expected):
    a=np.asarray(actual); b=np.asarray(expected)
    same_shape=a.shape==b.shape
    same_dtype=a.dtype==b.dtype
    same_values=same_shape and np.array_equal(a,b)
    item={'shape_actual':list(a.shape),'shape_expected':list(b.shape),'dtype_actual':str(a.dtype),'dtype_expected':str(b.dtype),'shape_equal':same_shape,'dtype_equal':same_dtype,'value_equal':bool(same_values)}
    if same_shape and a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu':
        aa=a.astype(np.float64); bb=b.astype(np.float64)
        item['max_abs_diff']=float(np.max(np.abs(aa-bb),initial=0.0))
        item['mismatch_count']=int(np.count_nonzero(a!=b))
    return item

def main():
    for key,path in [('base_candidate',BASE),('face_report',FACE_REPORT),('run003_refined',FROZEN),('registered_depths',DEPTH_OUTPUT)]:
        actual=sha(path)
        if actual != PINS[key]: raise RuntimeError(f'{key} pin mismatch {actual}')
    started=time.monotonic()
    base=np.load(BASE,allow_pickle=False); frozen=np.load(FROZEN,allow_pickle=False)
    fr=json.loads(FACE_REPORT.read_text())
    if not fr['complete'] or fr.get('geometry_or_native_admission') is not False: raise RuntimeError('face report incomplete or not diagnostic')
    rows=[]; pair_count=0
    for bank in fr['banks']:
        result=bank.get('result')
        if result is None: continue
        pairs=result.get('triangle_pairs') or []
        if int(result.get('count',-1)) != len(pairs): raise RuntimeError('hit-count mismatch')
        for pair in pairs:
            if len(pair)!=2: raise RuntimeError('malformed triangle pair')
            rows.append(int(pair[0])); pair_count+=1
    crossed=np.unique(np.asarray(rows,dtype=np.int64))
    if len(crossed)!=330 or pair_count!=1638: raise RuntimeError(f'unexpected crossing inventory: {len(crossed)} faces / {pair_count} pairs')
    owner=loadmod('owner_refinement_019',OWNER)
    result=owner.subdivide_crossed_faces_affine(
        base['points_world_m'],base['source_O_faces'],crossed,
        parent_face_markers=base['parent_source_face_marker'])
    comparisons={
        'points_world_m':compare('points_world_m',result.points,frozen['points_world_m']),
        'source_O_faces':compare('source_O_faces',result.faces,frozen['source_O_faces']),
        'refinement_face_parent_face_row':compare('refinement_face_parent_face_row',result.child_parent_face_rows,frozen['refinement_face_parent_face_row']),
        'refinement_face_parent_barycentric':compare('refinement_face_parent_barycentric',result.child_parent_face_barycentric,frozen['refinement_face_parent_barycentric']),
        'parent_source_face_marker':compare('parent_source_face_marker',result.child_parent_face_markers,frozen['parent_source_face_marker']),
        'refinement_node_kind':compare('refinement_node_kind',result.refinement_node_kind,frozen['refinement_node_kind']),
        'refinement_node_parent_edge_rows':compare('refinement_node_parent_edge_rows',result.refinement_node_parent_edge_rows,frozen['refinement_node_parent_edge_rows']),
        'refinement_node_parent_edge_weights':compare('refinement_node_parent_edge_weights',result.refinement_node_parent_edge_weights,frozen['refinement_node_parent_edge_weights']),
        'refinement_node_parent_face_indptr':compare('refinement_node_parent_face_indptr',result.refinement_node_parent_face_indptr,frozen['refinement_node_parent_face_indptr']),
        'refinement_node_parent_face_rows':compare('refinement_node_parent_face_rows',result.refinement_node_parent_face_rows,frozen['refinement_node_parent_face_rows']),
        'refinement_node_parent_face_barycentric':compare('refinement_node_parent_face_barycentric',result.refinement_node_parent_face_barycentric,frozen['refinement_node_parent_face_barycentric']),
        'source_prolongation_indptr':compare('source_prolongation_indptr',result.source_prolongation_indptr,frozen['source_prolongation_indptr']),
        'source_prolongation_node_rows':compare('source_prolongation_node_rows',result.source_prolongation_node_rows,frozen['source_prolongation_node_rows']),
        'source_prolongation_weights':compare('source_prolongation_weights',result.source_prolongation_weights,frozen['source_prolongation_weights']),
    }
    # Compare the owner helper's depth result with the separately frozen adapter output.
    adapter=loadmod('depth_adapter_018',ADAPTER)
    paths=adapter.DEFAULTS
    source_base=np.load(paths['base_candidate'],allow_pickle=False)
    fields=np.load(paths['field_asset'],allow_pickle=False)
    depth_owner=adapter._load_module('depth_owner_for_parity',paths['depth_owner'])
    assembler=adapter._load_module('layer_assembler_for_parity',paths['layer_assembler'])
    coarse_faces,_=assembler.face_csv(paths['coarse_faces'])
    source_depths=depth_owner.registered_depths(source_base,fields,coarse_faces)
    hsha=hashlib.sha256(np.ascontiguousarray(source_depths).tobytes()).hexdigest()
    owner_depths=owner.prolong_registered_depth_fields(
        source_depths,result.source_prolongation_indptr,result.source_prolongation_node_rows,
        result.source_prolongation_weights,expected_source_field_sha256=hsha,source_field_sha256=hsha)
    saved_depths=np.load(DEPTH_OUTPUT,allow_pickle=False)['registered_depths_m']
    comparisons['registered_depths_m']=compare('registered_depths_m',owner_depths,saved_depths)
    exact_fields=[k for k,v in comparisons.items() if v['value_equal']]
    report={
        'status':'CANONICAL_RUN003_PARITY_COMPLETE',
        'inputs_sha256':{k:sha(p) for k,p in [('base_candidate',BASE),('face_report',FACE_REPORT),('run003_refined',FROZEN),('registered_depths',DEPTH_OUTPUT),('canonical_owner',OWNER),('depth_adapter',ADAPTER)]},
        'crossing_inventory':{'all_bank_count':len(fr['banks']),'complete':fr['complete'],'triangle_bank_hit_records':pair_count,'unique_crossed_parent_faces':int(len(crossed))},
        'canonical_output':{'point_count':len(result.points),'face_count':len(result.faces),'selected_edge_count':len(result.selected_edges),'affected_parent_face_count':len(result.affected_parent_face_rows),'centroid_count':len(result.face_centroid_rows)},
        'comparisons':comparisons,
        'exact_value_equal_fields':exact_fields,
        'geometry_or_native_admission':False,
        'physical_steps':0,
        'elapsed_seconds':time.monotonic()-started,
        'notes':['Array ordering is checked as emitted; no ID reordering is applied.','Depth comparison uses owner 156 source fields and the frozen CSR; no mass transfer or geometry qualification is claimed.']
    }
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
