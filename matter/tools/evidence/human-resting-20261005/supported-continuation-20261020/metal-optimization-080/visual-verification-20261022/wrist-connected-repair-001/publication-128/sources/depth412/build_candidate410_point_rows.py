#!/usr/bin/env python3
"""Build candidate-matched 410 point-depth rows from its exact 185-bank query."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/Users/n/Work/numi-human-connected-surface-069')
TASK = ROOT / 'D/point-depth-barriers-412'
CANDIDATE = ROOT / 'D/postfit-backtrack-410/run-001/backtrack-373/pass-002/candidate.npz'
AUDIT = ROOT / 'D/postfit-backtrack-410/run-001/backtrack-373/pass-002'
POINT_SUMMARY = AUDIT / 'audit-130/summary.json'
SURFACE_REPORT = AUDIT / 'audit-159/report.json'
QUERY = AUDIT / 'audit-130/query'
RECEIPT = QUERY / 'query-receipt.json'
POINT_CSV, PAIRS_CSV, NODES_CSV = (QUERY / n for n in ('points.csv', 'pairs.csv', 'nodes.csv'))
THRESHOLD_M = 0.0011
OUTDIR = TASK / 'candidate-410-141ca8-near-001'
EXPECTED_CANDIDATE_SHA = '141ca8ff15a4377092e198c9d7d52d08034541cbf295979e7b539c34d508a090'
EXPECTED_POINT_SUMMARY_SHA = '44570bf22c5726985afe653fa6f12a8128bb4f6be13fcf233690faa47cb27c97'
EXPECTED_SURFACE_SHA = 'f8aed46baed83a7260ce12e874baedeb143e3bdf25be19798b2a94cfc83ab0e8'
HELPER = ROOT / 'D/point-depth-barriers-357/point_depth_barriers_357.py'
sys.path.insert(0, str(HELPER.parent))
from point_depth_barriers_357 import build_point_depth_rows

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rows(path: Path):
    with path.open('r',newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    if OUTDIR.exists(): raise FileExistsError(OUTDIR)
    pins={
      'candidate':(CANDIDATE,EXPECTED_CANDIDATE_SHA),
      'point_summary':(POINT_SUMMARY,EXPECTED_POINT_SUMMARY_SHA),
      'surface_report':(SURFACE_REPORT,EXPECTED_SURFACE_SHA),
      'helper':(HELPER,'dc8078515dd12b2edc62c0bc75759a7cb9472c9914e4704ecd41e04c4d5378e1'),
    }
    pin_hashes={}
    for label,(path,expected) in pins.items():
        got=sha(path)
        if got!=expected: raise ValueError(f'{label} SHA mismatch: {got}')
        pin_hashes[label]={'path':str(path),'sha256':got}
    point_summary=json.loads(POINT_SUMMARY.read_text())
    surface=json.loads(SURFACE_REPORT.read_text())
    if point_summary.get('candidate_sha256')!=EXPECTED_CANDIDATE_SHA or point_summary.get('points')!=20889 or point_summary.get('faces')!=41733 or point_summary.get('complete_surface_audit') is not True or point_summary.get('exact_PLC_surface_pairs')!=0 or point_summary.get('geometry_admitted') is not False or point_summary.get('accepted_physical_steps')!=0:
        raise ValueError('408 exact point/self summary is incomplete or candidate-mismatched')
    if surface.get('candidate_sha256')!=EXPECTED_CANDIDATE_SHA or surface.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT' or surface.get('complete') is not True or surface.get('bank_count')!=185 or surface.get('expected_bank_count')!=185 or surface.get('surface_point_count')!=20889 or surface.get('surface_face_count')!=41733:
        raise ValueError('408 exact surface audit is incomplete or candidate-mismatched')
    bank_keys=[(int(b['body']),int(b['stable_bone_id'])) for b in surface['banks']]
    if len(bank_keys)!=185 or len(set(bank_keys))!=185: raise ValueError('surface audit does not cover 185 unique typed banks')
    surface_pairs=sum(int((b.get('result') or {}).get('count',0)) for b in surface['banks'])
    if surface_pairs!=3268: raise ValueError(f'unexpected 408 surface-pair count {surface_pairs}')
    receipt=json.loads(RECEIPT.read_text())
    if int(receipt.get('query_rows',-1))!=20889 or int(receipt.get('physical_steps',-1))!=0 or receipt.get('identity_domain')!='mechanics_node_row':
        raise ValueError('410 query receipt identity/count/step mismatch')
    for p,key in ((POINT_CSV,'points_sha256'),(PAIRS_CSV,'pairs_sha256'),(NODES_CSV,'nodes_sha256')):
        if sha(p)!=receipt.get(key): raise ValueError(f'query receipt hash mismatch for {p.name}')
    argv=receipt.get('argv',[])
    if len(argv)<7 or int(argv[6])!=185: raise ValueError('query receipt does not pin 185 banks')
    body_csv=Path(argv[2])
    if not body_csv.is_file(): raise FileNotFoundError(body_csv)
    with np.load(CANDIDATE,allow_pickle=False) as z:
        points=z['points_world_m']
        faces=z['source_O_faces']
        if points.dtype!=np.float32 or points.shape!=(20889,3) or faces.shape!=(41733,3): raise ValueError('candidate shape/dtype mismatch')
        if sha_array(points)!=receipt.get('points_sha256'):
            # Query CSV and receipt hash the CSV bytes, while this array has a different serialization.
            qrows=rows(POINT_CSV)
            qpoints=np.asarray([[float(x['x_m']),float(x['y_m']),float(x['z_m'])] for x in qrows],dtype=np.float32)
            if qpoints.shape!=points.shape or not np.array_equal(qpoints,points): raise ValueError('query F32 points do not exactly match candidate')
        else:
            qpoints=points
    pair_rows=rows(PAIRS_CSV)
    selected_out=[]
    inside_count=0
    min_out=float('inf'); max_out=0.0
    for r in pair_rows:
        c=r['classification']
        if c=='inside':
            inside_count+=1
            if int(r['ambiguous'])!=0 or int(r['inside_rays'])<=0 or int(r['outside_rays'])!=0: raise ValueError('inconsistent inside row')
        elif c=='outside':
            d=float(r['nearest_distance_m'])
            if not np.isfinite(d) or d<0: raise ValueError('invalid outside distance')
            if d<=THRESHOLD_M:
                selected_out.append([int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id']),int(r['nearest_triangle'])])
                min_out=min(min_out,d); max_out=max(max_out,d)
        else: raise ValueError(f'unexpected query classification: {c}')
    if inside_count!=406: raise ValueError(f'expected 406 inside pair rows, got {inside_count}')
    if len({tuple(x) for x in selected_out})!=len(selected_out): raise ValueError('duplicate selected outside typed witness')
    inside_spec={'kind':'inside','points_csv':str(POINT_CSV),'pairs_csv':str(PAIRS_CSV),'nodes_csv':str(NODES_CSV),'query_receipt_json':str(RECEIPT),'require_points_match':True,'expected_selected_count':inside_count}
    outside_spec={'kind':'outside','points_csv':str(POINT_CSV),'pairs_csv':str(PAIRS_CSV),'nodes_csv':str(NODES_CSV),'query_receipt_json':str(RECEIPT),'require_points_match':True,'selected_typed_pairs':selected_out,'expected_selected_count':len(selected_out)}
    node_rows,normals,lower,typed,records,report=build_point_depth_rows(points,[inside_spec,outside_spec],body_csv,expected_point_count=20889,expected_bank_count=185,baseline_feasibility_tolerance_m=1e-10,reconstruction_tolerance_m=2e-8)
    if report['inside_row_count']!=406 or report['outside_row_count']!=len(selected_out) or report['maximum_zero_step_violation_m']!=0.0:
        raise ValueError('row builder baseline/selected count gate failed')
    if not all(q['complete_bank_accounting_verified'] and q['bank_count']==185 and q['query_rows']==20889 and q['ambiguous_rows']==0 and q['invalid_rows']==0 for q in report['query_sources']):
        raise ValueError('row helper lacks complete per-query 185-bank accounting')
    if len(node_rows)!=len(records) or normals.shape!=(len(records),3) or lower.shape!=(len(records),) or typed.shape!=(len(records),4): raise ValueError('invalid point-depth row array shapes')
    OUTDIR.mkdir(parents=True,exist_ok=False)
    npz=OUTDIR/'point-depth-rows.npz'
    np.savez_compressed(npz,node_rows=node_rows,normals_world_unit_covector=normals,lower_delta_m=lower,typed_ids=typed)
    report.update({
      'status':'PASS_CANDIDATE_MATCHED_410_POINT_DEPTH_ROWS',
      'candidate_npz':str(CANDIDATE),'candidate_sha256':EXPECTED_CANDIDATE_SHA,'candidate_points_sha256':sha_array(points),
      'point_summary_sha256':EXPECTED_POINT_SUMMARY_SHA,'surface_report_sha256':EXPECTED_SURFACE_SHA,'surface_pair_count':surface_pairs,
      'query_receipt_path':str(RECEIPT),'query_receipt_sha256':sha(RECEIPT),'query_points_csv_sha256':sha(POINT_CSV),
      'query_pairs_csv_sha256':sha(PAIRS_CSV),'query_nodes_csv_sha256':sha(NODES_CSV),'body_csv_sha256':sha(body_csv),
      'helper_path':str(HELPER),'helper_sha256':sha(HELPER),'output_npz':str(npz),'output_npz_sha256':sha(npz),
      'selection_policy':{'inside':'all exact inside typed rows','outside':'all exact outside typed rows with nearest_distance_m <= inclusive threshold','distance_metric':'raw bank-local Euclidean nearest distance','threshold_m':THRESHOLD_M,'inside_rows':inside_count,'outside_rows':len(selected_out),'outside_min_m':min_out,'outside_max_m':max_out},
      'input_pins':pin_hashes,'topology':[20889,41733],'typed_banks':185,'physical_steps':0,'geometry_admitted':False,
    })
    rp=OUTDIR/'report.json'; rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'output_npz':str(npz),'output_npz_sha256':sha(npz),'report_path':str(rp),'report_sha256':sha(rp),'rows':len(records),'inside':inside_count,'outside':len(selected_out),'unique_nodes':report['unique_node_count'],'zero_step_violation_m':report['maximum_zero_step_violation_m'],'candidate_sha256':EXPECTED_CANDIDATE_SHA,'query_receipt_sha256':sha(RECEIPT)},indent=2))
if __name__=='__main__': main()
