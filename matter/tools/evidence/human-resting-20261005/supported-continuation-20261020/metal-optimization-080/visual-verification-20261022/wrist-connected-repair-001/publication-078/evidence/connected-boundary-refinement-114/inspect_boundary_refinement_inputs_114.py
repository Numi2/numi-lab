from __future__ import annotations
import gzip, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/Users/numi/Work/numi-human-connected-surface-069')
OUT = ROOT / 'connected-boundary-refinement-114' / 'baseline-edge-incidence-114.json'
BASELINE = ROOT / 'publication-075/evidence/connected-cell-witness-103/run-001/baseline'
EDGE_REPORT = BASELINE / 'tet-edge-report.json.gz'
CANDIDATE_103 = BASELINE / 'candidate.npz'
COMPACT_084 = ROOT / 'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
MARKERS = ROOT / 'whole-reference-tetgen-084-bounded/facet-marker-map.csv'
PLC_BUILDER = ROOT / 'whole-reference-tetgen-084-bounded/build_whole_reference_plc_083.py'
TRANSFER_ADAPTER = ROOT / 'publication-072/evidence/transfer-adapter-075/load_transfer_075.py'
PATH_REPORT = ROOT / 'digit-source-diagnosis-110/report-002.json'
PATH_ROUTE_REPORT = ROOT / 'digit-source-diagnosis-110/report-004.json'
PATH_MIDPOINT = ROOT / 'digit-source-diagnosis-110/fine-path-midpoint-110.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    edge_report=json.load(gzip.open(EDGE_REPORT,'rt'))
    z=np.load(COMPACT_084,allow_pickle=False)
    O=np.asarray(z['source_O_faces'],dtype=np.int64)
    witness_groups=defaultdict(set)
    for w in edge_report['witnesses']:
        a,b=map(int,w['endpoint_local_nodes'])
        if a<2411 and b<2411:
            e=tuple(sorted((a,b)))
            witness_groups[e].add((int(w['body']),int(w['bone_stable_id'])))
    face_uses=defaultdict(list)
    for fi,f in enumerate(O):
        a,b,c=map(int,f)
        for u,v in ((a,b),(b,c),(c,a)):
            face_uses[tuple(sorted((u,v)))].append({'source_O_face_row':fi,'directed_edge':[u,v]})
    candidate_edges=sorted(witness_groups)
    eligible=[]; excluded=[]
    for e in candidate_edges:
        uses=face_uses.get(e,[])
        item={'edge':list(e),'baseline_bank_groups':len(witness_groups[e]),'incident_source_O_faces':uses}
        if len(uses)==2 and uses[0]['directed_edge']==uses[1]['directed_edge'][::-1]:
            item['source_parent_markers']=[100001+u['source_O_face_row'] for u in uses]
            item['eligible_for_conforming_O_subdivision']=True
            eligible.append(item)
        else:
            item['eligible_for_conforming_O_subdivision']=False
            item['reason']='not an interior two-face O edge with opposite oriented use'
            excluded.append(item)
    path_report=json.loads(PATH_REPORT.read_text()) if PATH_REPORT.exists() else None
    midpoint=json.loads(PATH_MIDPOINT.read_text()) if PATH_MIDPOINT.exists() else None
    report={
      'schema':'numi.human.connected_boundary_refinement_input_audit.v1',
      'status':'TOPOLOGY_ONLY_SOURCE_MIDPOINTS_INCOMPLETE_NOT_ADMITTED',
      'inputs':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (EDGE_REPORT,CANDIDATE_103,COMPACT_084,MARKERS,PLC_BUILDER,TRANSFER_ADAPTER,PATH_REPORT,PATH_MIDPOINT)},
      'counts':{'baseline103_O_layer_edge_hit_records':sum(len(v) for v in witness_groups.values()),'distinct_candidate_edges':len(candidate_edges),'distinct_candidate_edge_bank_groups':sum(len(v) for v in witness_groups.values()),'eligible_interior_oriented_O_edges':len(eligible),'eligible_O_edge_bank_groups':sum(len(witness_groups[tuple(e['edge'])]) for e in eligible),'excluded_non_O_or_nonmanifold_edges':len(excluded)},
      'source_O_face_domain':{'rows':int(len(O)),'unique_vertices':int(len(np.unique(O))),'marker_rule_from_083':'marker=100000+source_O_face_row+1'},
      'eligible_edges':eligible,
      'excluded_edges':excluded,
      'single_source_path_example':None if midpoint is None else {
        'edge':midpoint['coarse_O_edge'],'midpoint_world_m':midpoint['detailed_path']['midpoint_world_m'],
        'detailed_path_vertex_ids':[r['detail_row'] for r in midpoint['detailed_path']['edge_endpoint_records']],
        'source_face_rows':midpoint['parent_face_lineage'],
        'detail_faces_at_midpoint':midpoint['detailed_path']['incident_NHSKIN_triangle_rows'],
        'status':'one deterministic path witness only; report-002 does not establish path uniqueness or source-normal compatibility; not accepted as an insertion sample'},
      'source_path_segment_audit':{'report_path':str(PATH_ROUTE_REPORT),'sha256':sha(PATH_ROUTE_REPORT) if PATH_ROUTE_REPORT.exists() else None,'interpretation':'the 329-340 detail path segments were exact-disjoint from the two named emitted banks in report-004; this is a bounded two-bank path audit, not source-path uniqueness, normal compatibility, or all-185-bank clearance'},
      'limitations':['No O midpoint sample set exists here for all 145 eligible edges.','The 110 sample for edge (329,340) is a deterministic detailed NHSKIN path witness, but its report leaves path uniqueness and source-normal compatibility unresolved.','No surface refinement candidate, TetGen run, collision clearance, conservative transfer, or physical/native admission was produced.']}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(OUT),'sha256':sha(OUT),'counts':report['counts'],'excluded':excluded,'example':report['single_source_path_example']},indent=2,sort_keys=True))
if __name__=='__main__': main()
