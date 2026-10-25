#!/Users/n/numi-human-prep-venv-20261005/bin/python
"""Saved-witness diagnosis of O/I/D tetrahedral edge hits near four digit columns.

Read-only: uses exact baseline-103 edge/bank witnesses and the frozen 084/D75
coarse-to-detailed surface transfer. It does not launch a geometry query or
modify any candidate.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, struct
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

WORK = Path('/Users/n/Work/numi-human-connected-surface-069')
BASE = WORK / 'connected-cell-witness-103/run-001/baseline'
COMPACT = WORK / 'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
TRANSFER = WORK / 'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002'
MARKER_MAP = WORK / 'whole-reference-tetgen-084-bounded/facet-marker-map.csv'
DETAIL = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bilateral-hand-functional-rebind-020/run-001/candidate.nhskin')
OUT = WORK / 'digit-source-diagnosis-110'
COLS = (328, 340, 2277, 2301)
COLUMN_COUNT = 2411
EXPECTED = {
    'baseline_points': 'f15ea806a61c6a70fb4d051c932df22ac173b1f39e3f60662fad396362a8b58d',
    'baseline_pairs': '95023ef49a865473ab8baaca65da37bc45d78d5b4adf9e16ee7effa604bad4eb',
    'baseline_nodes': 'e49ef0e61ed9551bc3782e3bb8def6a45b5b2b32124ce1c7b7dabb9e286badef',
    'compact': '5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
    'detail': '7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb',
    'transfer': '55329313bf9c2c3b613fcd7eb52dd9b7b453520dd1b4a14e0e0d6ac457ce8519',
    'edge_report': '41c8b9625e99ae6f5a4b4f3d909e847c906c53adc7523b221e0f2373faa7c2ee',
}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def check_pin(name: str, path: Path) -> str:
    got=sha(path)
    want=EXPECTED[name]
    if want.startswith('frozen_'):
        raise RuntimeError(f'fill exact {name} pin before execution')
    if got != want: raise RuntimeError(f'{name} hash mismatch: {got} != {want}')
    return got

def load_csv_gz(path: Path):
    with gzip.open(path, 'rt', newline='') as f:
        return list(csv.DictReader(f))

def layer_for_source_row(srcrow: int):
    if srcrow < 0: return 'generated'
    if srcrow < COLUMN_COUNT: return 'O'
    if srcrow < 2*COLUMN_COUNT: return 'I'
    if srcrow < 3*COLUMN_COUNT: return 'D'
    raise RuntimeError(f'unknown source row {srcrow}')

def layer_pair(u: int, v: int, srcrows: np.ndarray):
    lu,lv=layer_for_source_row(int(srcrows[u])),layer_for_source_row(int(srcrows[v]))
    if 'generated' in (lu,lv): return 'generated'
    order={'O':0,'I':1,'D':2}
    a,b=sorted((lu,lv),key=order.get)
    return f'{a}-{b}'

def parse_nhskin_faces(path: Path):
    raw=path.read_bytes()
    if len(raw)<60: raise RuntimeError('truncated NHSKIN')
    magic,abi,nb,nv,ni,_=struct.unpack_from('<8s5I',raw)
    if magic != b'NHSKIN1\0' or abi != 5 or ni%3: raise RuntimeError('unexpected NHSKIN ABI')
    io=60+nb*36+nv*56
    wo=io+ni*4
    if len(raw) != wo+nv*nb*4: raise RuntimeError('NHSKIN length/header mismatch')
    faces=np.frombuffer(raw,dtype='<u4',count=ni,offset=io).reshape(-1,3).copy()
    return faces, {'magic':magic.decode('ascii','replace').rstrip('\0'),'abi':int(abi),'bindings':int(nb),'vertices':int(nv),'indices':int(ni)}

def make_detail_edge_support_map(detail_faces: np.ndarray, W: csr_matrix, o_surface_edges: set[tuple[int,int]], support_tol=1e-10):
    """Return O coarse edge -> exact detailed-face-edge count under frozen W.

    A detailed mesh edge maps to an O coarse edge only when every non-negligible
    transfer coefficient on its endpoints is supported by the same two coarse
    O vertices. This identifies topology represented on the same O-edge; it is
    not a claim that a coarse tet chord is itself a physical skin edge.
    """
    de=np.concatenate((detail_faces[:,[0,1]],detail_faces[:,[1,2]],detail_faces[:,[2,0]]),axis=0)
    de=np.unique(np.sort(de,axis=1),axis=0)
    mapped=Counter()
    skipped_inactive=0
    mapped_edge_count=0
    for a,b in de:
        ra0,ra1=int(W.indptr[a]),int(W.indptr[a+1])
        rb0,rb1=int(W.indptr[b]),int(W.indptr[b+1])
        if ra0==ra1 or rb0==rb1:
            skipped_inactive+=1; continue
        sa={int(i) for i,x in zip(W.indices[ra0:ra1],W.data[ra0:ra1]) if abs(float(x))>support_tol}
        sb={int(i) for i,x in zip(W.indices[rb0:rb1],W.data[rb0:rb1]) if abs(float(x))>support_tol}
        su=sa|sb
        if len(su)!=2: continue
        uv=tuple(sorted(su))
        if uv in o_surface_edges:
            mapped[uv]+=1; mapped_edge_count+=1
    return mapped, {'unique_detail_edges':int(len(de)),'inactive_support_edges':skipped_inactive,
                    'detail_edges_mapped_to_coarse_O_edges':mapped_edge_count,
                    'coarse_O_edges_with_detail_edge_support':len(mapped),
                    'support_threshold_abs_weight':support_tol}

# Verify exact immutable inputs before analyzing.
points_path=BASE/'query/points.csv'
pairs_path=BASE/'query/pairs.csv'
nodes_path=BASE/'query/nodes.csv'
edge_path=BASE/'tet-edge-report.json'
transfer_path=TRANSFER/'Wnew-detail-to-084-nodes.npz'
for n,p in [('baseline_points',points_path),('baseline_pairs',pairs_path),('baseline_nodes',nodes_path),('compact',COMPACT),('detail',DETAIL),('transfer',transfer_path),('edge_report',edge_path)]:
    check_pin(n,p)

with edge_path.open('rt') as f: edge=json.load(f)
with np.load(COMPACT,allow_pickle=False) as z:
    ref=np.asarray(z['reference_points_m'],dtype=np.float64)
    current=np.asarray(z['current037_points_m_f32'])
    srcrows=np.asarray(z['source_input_row'],dtype=np.int64)
    tet=np.asarray(z['tetrahedra_zero_based'],dtype=np.int64)
    sf=np.asarray(z['surface_faces_zero_based'],dtype=np.int64)
    markers=np.asarray(z['surface_face_markers'],dtype=np.int64)
    original_O=np.asarray(z['current037_points_m_f32'][:COLUMN_COUNT],dtype=np.float64)
    global_ids=np.asarray(z['source_global_node_ids'],dtype=np.int64)
with np.load(BASE/'candidate.npz',allow_pickle=False) as z:
    query_points=np.asarray(z['points_world_f32'])
if query_points.shape != current.shape or query_points.dtype != np.float32 or current.dtype != np.float32:
    raise RuntimeError('query/compact positions have unexpected shape or dtype')
if not np.array_equal(query_points,current): raise RuntimeError('baseline point query differs from frozen compact current positions')
if len(srcrows)!=len(query_points) or len(srcrows)!=7360: raise RuntimeError('unexpected output node domain')
if not np.array_equal(srcrows[:7233],np.arange(7233)) or np.count_nonzero(srcrows<0)!=127:
    raise RuntimeError('source input/output row mapping differs from frozen compact schema')

# Build the actual compacted O shell edge set from tagged outer faces.
with MARKER_MAP.open(newline='') as f: marker_surface={int(r['marker']):r['surface'] for r in csv.DictReader(f)}
o_face_ids=np.asarray([i for i,m in enumerate(markers) if marker_surface.get(int(m))=='O'],dtype=np.int64)
o_faces=sf[o_face_ids]
o_surface_edges=set()
for tri in o_faces:
    for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
        o_surface_edges.add(tuple(sorted((int(a),int(b)))) )

# Load actual frozen detailed NHSKIN topology and its exact accepted W map.
detail_faces,detail_header=parse_nhskin_faces(DETAIL)
wf=np.load(transfer_path,allow_pickle=False)
W=csr_matrix((wf['data'],wf['indices'],wf['indptr']),shape=tuple(int(x) for x in wf['shape']))
if W.shape != (detail_header['vertices'],len(current)): raise RuntimeError('transfer shape does not match NHSKIN/coarse node domains')
if detail_faces.max()>=W.shape[0]: raise RuntimeError('detailed face index exceeds W rows')
detail_o_edge_counts,detail_edge_stats=make_detail_edge_support_map(detail_faces,W,o_surface_edges)

# Deduplicate per edge/bank; retain all exact triangle witnesses in each group.
groups=defaultdict(list)
for w in edge['witnesses']:
    u,v=map(int,w['endpoint_query_rows'])
    if not (0<=u<len(srcrows) and 0<=v<len(srcrows)): raise RuntimeError('edge endpoint outside baseline query rows')
    e=tuple(sorted((u,v)))
    bank=(int(w['body']),int(w['bone_stable_id']))
    groups[(e,bank)].append(w)
layer_counts=defaultdict(lambda:{'edge_bank_groups':0,'unique_edges':set(),'banks':set(),'triangle_hit_witnesses':0,'outer_O_edges':0,'O_internal_chords':0,'O_edges_with_detailed_edge_support':0,'O_edges_without_detailed_edge_support':0})
for (e,bank),ws in groups.items():
    cat=layer_pair(e[0],e[1],srcrows)
    x=layer_counts[cat]
    x['edge_bank_groups']+=1; x['unique_edges'].add(e); x['banks'].add(bank); x['triangle_hit_witnesses']+=len(ws)
    if cat=='O-O':
        if e in o_surface_edges:
            x['outer_O_edges']+=1
            if detail_o_edge_counts.get(e,0): x['O_edges_with_detailed_edge_support']+=1
            else: x['O_edges_without_detailed_edge_support']+=1
        else: x['O_internal_chords']+=1
layer_summary={}
for cat,x in sorted(layer_counts.items()):
    layer_summary[cat]={k:(len(v) if isinstance(v,set) else v) for k,v in x.items()}

# Detailed evidence for all baseline crossing groups incident to requested columns.
def endpoint_record(row,target_col):
    s=int(srcrows[row]); lay=layer_for_source_row(s)
    p=query_points[row].astype(np.float64)
    own_col=(s%COLUMN_COUNT) if s>=0 else None
    own_o=original_O[own_col] if own_col is not None else None
    return {'query_row':int(row),'source_input_row':s,'source_layer':lay,
            'source_column':int(own_col) if own_col is not None else None,
            'world_current037_f32_m':[float(q) for q in p],
            'distance_to_requested_original_O_mm':float(np.linalg.norm(p-original_O[target_col])*1000.0),
            'distance_to_own_column_original_O_mm':(float(np.linalg.norm(p-own_o)*1000.0) if own_o is not None else None)}

def col_rows(c): return {c,c+COLUMN_COUNT,c+2*COLUMN_COUNT}
selected={}
for c in COLS:
    rows=col_rows(c)
    recs=[]
    for (e,bank),ws in groups.items():
        if rows.isdisjoint(e): continue
        u,v=e
        cat=layer_pair(u,v,srcrows)
        d={'edge_query_rows':[u,v],'layer_pair':cat,
           'endpoints':[endpoint_record(u,c),endpoint_record(v,c)],
           'bank_body':bank[0],'bone_stable_id':bank[1],
           'triangle_witness_count':len(ws),
           'triangle_stable_ids':sorted({int(w['triangle_stable_id']) for w in ws}),
           'hit_parameters_t':sorted(float(w['exact_result']['t']) for w in ws),
           'exact_classifications':sorted({str(w['exact_result']['classification']) for w in ws})}
        if cat=='O-O':
            d['outer_surface_classification']='actual_compacted_O_surface_edge' if e in o_surface_edges else 'internal_volume_chord_not_on_O_surface'
            d['detailed_NHSKIN_edges_mapped_to_same_O_edge']=int(detail_o_edge_counts.get(e,0))
        recs.append(d)
    recs.sort(key=lambda q:(q['edge_query_rows'],q['bank_body'],q['bone_stable_id']))
    selected[str(c)]={'incident_source_rows':sorted(rows),'incident_edge_bank_group_count':len(recs),
                       'category_counts':dict(sorted(Counter(r['layer_pair'] for r in recs).items())),
                       'groups':recs}

# Summarize current cross-edge witnesses on requested columns by endpoint level.
# This is a source/topology diagnosis only; no inference that points are anatomically correct.
report={
 'status':'READ_ONLY_SAVED_WITNESS_DIAGNOSIS',
 'scope':'Baseline103 exact tetrahedral edge/bank witnesses; 084 output layers and outer-surface topology; frozen 075 detailed NHSKIN transfer graph. No query, geometry edit, or physical admission.',
 'input_pins':{
   'baseline_query_points_sha256':sha(points_path),'baseline_query_pairs_sha256':sha(pairs_path),'baseline_query_nodes_sha256':sha(nodes_path),
   'baseline_exact_edge_report_sha256':sha(edge_path),'baseline_compact_query_candidate_sha256':sha(BASE/'candidate.npz'),
   'compact_084_mesh_sha256':sha(COMPACT),'facet_marker_map_sha256':sha(MARKER_MAP),
   'detail_NHSKIN_sha256':sha(DETAIL),'Wnew_transfer_sha256':sha(transfer_path),
   'detail_face_source':'NHSKIN v5 ABI5 index payload parsed read-only; exact source payload bytes hash-pinned.'},
 'mapping_checks':{
   'baseline_query_rows':int(len(query_points)),'compact_output_rows':int(len(current)),
   'baseline_query_points_equal_compact_current037_F32_bitwise':True,
   'source_input_rows_0_through_7232_are_output_row_identity':True,
   'generated_output_rows':np.flatnonzero(srcrows<0).astype(int).tolist(),
   'query_positions_vs_reference_max_abs_m':float(np.max(np.abs(query_points.astype(np.float64)-ref))),
   'detail_header':detail_header,'transfer_shape':[int(W.shape[0]),int(W.shape[1])],
   'detail_to_coarse_W_nnz':int(W.nnz),'detail_to_coarse_edge_mapping':detail_edge_stats,
   'O_surface_face_count':int(len(o_faces)),'O_surface_unique_edges':int(len(o_surface_edges))},
 'exact_edge_report_counts':edge['counts'],
 'unique_edge_bank_groups_by_layer_pair':layer_summary,
 'O_O_interpretation':{
   'note':'O-O means both endpoints are original outer-layer source nodes. It does not itself imply a skin edge.',
   'outer_shell_edge_groups':layer_summary.get('O-O',{}).get('outer_O_edges',0),
   'internal_volume_chord_groups':layer_summary.get('O-O',{}).get('O_internal_chords',0),
   'outer_shell_edges_with_a_detailed_NHSKIN_edge_mapped_to_same_coarse_O_edge':layer_summary.get('O-O',{}).get('O_edges_with_detailed_edge_support',0),
   'outer_shell_edges_without_a_detailed_NHSKIN_edge_mapped_to_same_coarse_O_edge':layer_summary.get('O-O',{}).get('O_edges_without_detailed_edge_support',0),
   'interpretation_limit':'A tetrahedral O-O chord absent from the O boundary graph is an internal volume chord. A matching W-supported detailed edge is direct topology evidence that the coarse O surface edge is represented in the frozen detailed mesh; it does not establish an anatomical defect or its cause.'},
 'requested_columns':selected,
 'limitations':['The edge witness report counts exact segment-triangle hits; this is not a full tet-solid or physical-material clearance certificate.',
                'Generated endpoints are grouped as generated; this report does not assign them an O/I/D column unless their source_input_row is original.',
                'Detailed edge correspondence is inferred from the frozen W transfer support plus NHSKIN face adjacency. It is not a new geometric intersection audit.',
                'No source geometry, candidate, rest positions, weights, bindings, masses, or native state were changed.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(OUT/'report.json'),'groups':len(groups),'layer_summary':{k:{kk:vv for kk,vv in v.items()} for k,v in layer_summary.items()},'selected_counts':{k:v['incident_edge_bank_group_count'] for k,v in selected.items()},'O_O':report['O_O_interpretation']},indent=2))
