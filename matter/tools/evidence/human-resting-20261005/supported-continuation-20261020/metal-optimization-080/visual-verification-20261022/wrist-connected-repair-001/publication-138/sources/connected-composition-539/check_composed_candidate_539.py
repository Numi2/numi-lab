#!/usr/bin/env python3
"""Recompute construction arithmetic gates for the pinned 538 mesh and 536 points.

This is a read-only arithmetic receipt. It does not run geometry audits or admit the asset.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

D = Path('/Users/numi/Work/numi-human-connected-surface-069')
BASE = D/'D/finite-triangle-backtrack-512/all-pairs-001/backtrack-373/pass-008/candidate.npz'
COORD = D/'D/connected-composition-536/run-001/candidate.npz'
COORD_REPORT = D/'D/connected-composition-536/run-001/report.json'
COORD_AUDIT = D/'D/connected-composition-536/root-exact-run001/audit-130/summary.json'
COORD_PAIRS = D/'D/connected-composition-536/root-exact-run001/audit-130/surface-self-pairs.npy'
CAND = D/'D/connected-quality-patch-538/run-001/candidate.npz'
CAND_REPORT = D/'D/connected-quality-patch-538/run-001/report.json'
CAND_SCRIPT = D/'D/connected-quality-patch-538/compose_surviving_flips_538.py'
AUDIT130 = D/'D/connected-quality-patch-538/root-exact-run001/audit-130/summary.json'
AUDIT159 = D/'D/connected-quality-patch-538/root-exact-run001/audit-159/report.json'
FIT = D/'barycentric-connected-clearance-170/connected-local-fit-529/current-scaled-001/candidate.npz'
OWNER = D/'D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
AREA_OWNER = D/'barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/area_scp_support_023.py'
SOURCE_BASE = D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'

PINS = {
    'base_candidate': (BASE, 'c1c0b63ee448703ef3240dbcc251e650022d7f76f475bc75db9cae040d1e3a6c'),
    '536_candidate': (COORD, 'e880f1c96e8bb540f1918673fecd07e9dccfea54ebe5cefc91fc14a77307edc5'),
    '536_report': (COORD_REPORT, '5cee823ba93dbd105d4d8c2ca85821169564869a16518f9df43fa6102ab8d9bf'),
    '536_audit130': (COORD_AUDIT, 'b6141b11fb4043aff9151566ae56a9670540af183b9d1eda88522b1ab22b6380'),
    '536_self_pairs': (COORD_PAIRS, 'e9ed0f4b155ce3ec034478a693ac659829445e6c7eb57012bd4f4096b6c17ffc'),
    '538_candidate': (CAND, '2f52328e1105dad576717eae872391324a1172ceec3f74f23fae591891078c6e'),
    '538_report': (CAND_REPORT, 'b3fb7a298a284c158d4c6bb94e791230bed8c4f511f9561c45d665ad1de9c91a'),
    '538_composer': (CAND_SCRIPT, '9dc78be3a52e2607771c49b11b30e175099ae0c6295ccecf504cdf9c9519cd52'),
    '536_composer': (D/'D/connected-composition-536/compose_checked_topology_coordinates_536.py', 'd9367fca00cb7a42605a605cb0959fa002361196d312a0c772f592fc0763f38e'),
    'audit130_owner': (D/'audit_connected_outer_geometry_130.py', '52d45f287162d22dca417793c1393bcfbbe506c5ce85b828f009b4fd7ccd94b9'),
    'audit159_owner': (D/'audit_skin_bone_surfaces_159.py', 'de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1'),
    '529_fit': (FIT, '84832ab3222b7d9eb851a9e7d6b966c66e9e8c13994292c90b23e588218e2963'),
    '408_owner': (OWNER, '5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2'),
    'source_base_R0': (SOURCE_BASE, '70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31'),
}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()

def same(a,b):
    a=np.ascontiguousarray(a); b=np.ascontiguousarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def read_json(p): return json.loads(Path(p).read_text())
def load_npz(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot load {p}')
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def main():
    checker_path=Path(__file__).resolve()
    checker_sha_start=sha(checker_path)
    # Fail closed on every immutable owner/input. The two live audit receipts are
    # pinned by the output receipt after checking their candidate identities.
    for name,(p,want) in PINS.items():
        if want.endswith('pending'):
            continue
        if not p.is_file() or sha(p)!=want:
            raise ValueError(f'{name} source SHA mismatch or missing: {p}')
    base=load_npz(BASE); coord=load_npz(COORD); cand=load_npz(CAND); fit=load_npz(FIT)
    cr=read_json(CAND_REPORT); c536=read_json(COORD_REPORT)
    if cr.get('candidate_sha256') != sha(CAND) or cr.get('candidate_sha256') != PINS['538_candidate'][1]:
        raise ValueError('538 report/candidate identity mismatch')
    if c536.get('candidate_sha256') != sha(COORD) or c536.get('coordinate_branch_noncoordinate_arrays_bitwise_equal_to_base') is not True:
        raise ValueError('536 coordinate receipt does not match the pinned point field')
    if not same(cand['points_world_m'],coord['points_world_m']):
        raise ValueError('538 coordinates are not bitwise copied from 536')
    points=np.asarray(cand['points_world_m'])
    base_points=np.asarray(base['points_world_m'])
    if points.dtype != np.float32 or points.shape != base_points.shape or not np.isfinite(points).all():
        raise ValueError('candidate point field must be finite emitted Float32 with baseline shape')
    if base_points.dtype != np.float32 or not np.isfinite(base_points).all():
        raise ValueError('baseline point field is invalid')
    anchors=np.asarray(base['typed_coarse_anchor_output_rows_018'])
    if anchors.ndim!=1 or anchors.dtype.kind not in 'iu' or np.any(anchors<0) or np.any(anchors>=len(points)):
        raise ValueError('invalid typed anchor rows')
    anchor_pass=same(points[anchors],base_points[anchors])
    if not anchor_pass: raise ValueError('typed anchors changed')
    support=np.asarray(fit['local_witness_cluster_222_support_rows'])
    if support.ndim!=1 or support.dtype.kind not in 'iu' or len(np.unique(support))!=len(support) or np.any(support<0) or np.any(support>=len(points)):
        raise ValueError('invalid pinned 529 fit support')
    outside=np.ones(len(points),dtype=bool); outside[support]=False
    outside_unchanged=same(points[outside],base_points[outside])
    if not outside_unchanged: raise ValueError('coordinates changed outside 529 fit support')
    faces=np.asarray(cand['source_O_faces'])
    fixed_faces=np.asarray(cand['area_reference_037_source_O_faces'])
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or not same(faces,fixed_faces):
        raise ValueError('new topology and fixed-037 topology are mismatched')
    ref037=np.asarray(cand['area_reference_037_points_world_m_f64'])
    if ref037.dtype!=np.float64 or not np.isfinite(ref037).all():
        raise ValueError('fixed-037 reference coordinates invalid')
    owner=load_module('owner408_539',OWNER)
    # Full emitted-F32 area checks use current c1 coordinates, 538 faces, fixed-037 reference.
    area=owner.full_f32_area_gate(points,base_points.astype(np.float64)*1000.0,ref037*1000.0,faces,0.1)
    if area.get('pass') is not True: raise ValueError('full-face Float32 area gate failed')
    # 408 lower-form edge inequalities are evaluated on world-frame millimetres.
    A,rhs,edge_receipt=owner.build_edge_gradient_constraints(base_points.astype(np.float64)*1000.0,faces,support,0.5)
    delta=(points.astype(np.float64)-base_points.astype(np.float64))*1000.0
    slack=np.asarray(A@delta[support].reshape(-1)).ravel()-rhs
    edge_min=float(slack.min(initial=np.inf))
    if not np.isfinite(slack).all() or edge_min < -1e-7: raise ValueError('408 edge-gradient gate failed')
    step=np.linalg.norm(delta,axis=1)
    max_step=float(step.max(initial=0.0))
    if not np.isfinite(step).all() or max_step>1.0+1e-6: raise ValueError('1mm trust gate failed')
    # Keep the actual CSR row receipts and pin the source endpoint file used to form R0.
    with np.load(BASE,allow_pickle=False) as base_archive:
        source,source_levels=owner.compose_source_reference(base_archive,SOURCE_BASE)
    if not np.isfinite(source).all() or source.shape!=points.shape: raise ValueError('composed R0 invalid')
    if not source_levels or source_levels[-1].get('sha256')!=PINS['source_base_R0'][1]:
        raise ValueError('R0 source endpoint receipt mismatch')
    source_mm=np.linalg.norm(points.astype(np.float64)-source,axis=1)*1000.0
    source_max=float(source_mm.max(initial=0.0))
    if not np.isfinite(source_mm).all() or source_max>12.0+1e-6: raise ValueError('12mm source cap gate failed')
    # Validate real saved exact-audit receipts if available. Never run audits here.
    a130=read_json(AUDIT130); a159=read_json(AUDIT159)
    if a130.get('candidate_sha256')!=sha(CAND) or a130.get('complete_surface_audit') is not True:
        raise ValueError('538 audit-130 receipt incomplete or mismatched')
    bank_rows=a159.get('banks',[])
    bank_keys={(int(b['body']),int(b['stable_bone_id'])) for b in bank_rows}
    if a159.get('candidate_sha256')!=sha(CAND) or a159.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT' or a159.get('complete') is not True or a159.get('bank_count')!=a159.get('expected_bank_count') or a159.get('expected_bank_count')!=185 or len(bank_rows)!=185 or len(bank_keys)!=185:
        raise ValueError('538 audit-159 receipt incomplete or mismatched')
    surface_pairs=0
    for bank in bank_rows:
        result_row=bank.get('result')
        if result_row is None:
            if int(bank.get('skin_aabb_active',0))>0 and int(bank.get('bank_aabb_active',0))>0:
                raise ValueError('audit-159 missing result for active bank pair')
            continue
        count=int(result_row.get('count',-1)); pairs=result_row.get('triangle_pairs')
        if count<0 or not isinstance(pairs,list) or len(pairs)!=count:
            raise ValueError('audit-159 pair table/count mismatch')
        surface_pairs+=count
    self_pairs=np.load(COORD_PAIRS,allow_pickle=False)
    if self_pairs.ndim!=2 or self_pairs.shape[1]!=2 or self_pairs.dtype.kind not in 'iu':
        raise ValueError('536 exact self-pair array has invalid shape/type')
    if len(self_pairs)!=int(read_json(COORD_AUDIT).get('exact_PLC_surface_pairs',-1)):
        raise ValueError('536 self-pair table/report count mismatch')
    self_pairs_538_path=AUDIT130.parent/'surface-self-pairs.npy'
    self_pairs_538=np.load(self_pairs_538_path,allow_pickle=False)
    if self_pairs_538.ndim!=2 or self_pairs_538.shape[1]!=2 or self_pairs_538.dtype.kind not in 'iu':
        raise ValueError('538 exact self-pair array has invalid shape/type')
    self_zero=(len(self_pairs_538)==0 and int(a130.get('exact_PLC_surface_pairs',-1))==0)
    if len(self_pairs_538)!=int(a130.get('exact_PLC_surface_pairs',-1)):
        raise ValueError('538 self-pair table/report count mismatch')
    pins={name:{'path':str(p),'sha256':sha(p)} for name,(p,_) in PINS.items()}
    pins.update({'538_audit130':{'path':str(AUDIT130),'sha256':sha(AUDIT130)},
                 '538_audit159':{'path':str(AUDIT159),'sha256':sha(AUDIT159)},
                 '538_self_pair_table':{'path':str(self_pairs_538_path),'sha256':sha(self_pairs_538_path)},
                 '539_checker':{'path':str(checker_path),'sha256':checker_sha_start}})
    if any(sha(p)!=want for p,want in PINS.values()) or sha(checker_path)!=checker_sha_start:
        raise ValueError('pinned inputs or checker changed during arithmetic review')
    passed=bool(area['pass'] and edge_min>=-1e-7 and max_step<=1.0+1e-6 and source_max<=12.0+1e-6 and anchor_pass and outside_unchanged and self_zero)
    result={
      'schema':'numi.human.connected-composition-arithmetic-539.v1',
      'status':'ARITHMETIC_GATES_PASS_EXACT_SELF_ZERO_RECORDED_UNADMITTED' if passed else 'ARITHMETIC_GATE_OR_EXACT_SELF_FAILURE',
      'classification':'construction-only arithmetic verification; no geometry audit run by 539',
      'candidate_path':str(CAND),'candidate_sha256':sha(CAND),'source_pins':pins,
      'coordinate_field_bitwise_from_536':True,'anchor_rows':int(len(anchors)),
      'anchors_bitwise_unchanged':anchor_pass,'fit_support_rows':int(len(support)),
      'outside_fit_support_bitwise_unchanged':outside_unchanged,
      'maximum_step_mm':max_step,'trust_mm':1.0,'trust_pass':max_step<=1.0+1e-6,
      'source_cap_mm':12.0,'maximum_source_distance_mm':source_max,'source_cap_pass':source_max<=12.0+1e-6,
      'source_reference_R0':source_levels,
      'full_f32_area_gate':area,
      'edge_gradient':{'pass':edge_min>=-1e-7,'minimum_slack_mm':edge_min,'tolerance_mm':1e-7,'build_receipt':edge_receipt},
      'audit130_receipt':{'sha256':sha(AUDIT130),'complete':a130.get('complete_surface_audit'),'exact_self_pairs':a130.get('exact_PLC_surface_pairs'),'candidate_sha256':a130.get('candidate_sha256'),'geometry_admitted':False},
      'audit159_receipt':{'sha256':sha(AUDIT159),'complete':a159.get('complete'),'bank_count':a159.get('bank_count'),'expected_bank_count':a159.get('expected_bank_count'),'surface_triangle_crossing_pairs':surface_pairs,'candidate_sha256':a159.get('candidate_sha256'),'geometry_or_native_admission':False},
      'exact_global_self_zero':self_zero,'exact_self_pairs_538':int(len(self_pairs_538)),
      'exact_inside_point_pairs_538':int(a130.get('inside_node_bank_pairs',-1)),
      'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,
      'old_self_seed_or_313_linear_rows_revalidated_for_new_topology':False,
      'old_self_seed_or_313_rows_claimed_valid_for_new_topology':False,
      '539_arithmetic_uses_only_fresh_408_area_and_edge_calculations':True,
      'note':'Exact self/bone results are recorded as evidence only; this receipt does not certify clearance or physical transfer.'}
    od=D/'D/connected-composition-539'; od.mkdir(parents=True,exist_ok=True)
    dest=od/'run-001-arithmetic-receipt.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'receipt':str(dest),'candidate_sha256':result['candidate_sha256'],'status':result['status'],'maximum_step_mm':max_step,'maximum_source_distance_mm':source_max,'min_edge_slack_mm':edge_min,'audit130_self_pairs':a130.get('exact_PLC_surface_pairs'),'audit159_banks':a159.get('bank_count')}))
    if not passed: raise SystemExit(1)

if __name__=='__main__': main()
