#!/usr/bin/env python3
"""Replay arithmetic-only 458 C/h objective and F32 component-backtrack check.
No fit, geometry query, audit, or mutation of candidate data is performed.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

ROOT = Path('/Users/numi/Work/numi-human-connected-surface-069')
D = ROOT / 'D/surface-clearance-546'
P = D / 'connected-batch-619/run-006-from782-v8/round-001'
BASE = D / 'connected-topology-580/run-008-from796/lineage/candidate.npz'
AUDIT = D / 'connected-topology-580/run-008-from796/full-check/audit-159/report.json'
OWNER = ROOT / 'D/D/fixed-bone-outward-plane-458/fixed_bone_outward_plane_458.py'
FIT = P / 'fit-000/report.json'
TARGET_REPORT = P / 'prepared/surface-targets/report.json'
TARGETS = P / 'prepared/surface-targets/targets.npz'
SAT = P / 'fit-000.sat-crossing-inputs-458.npz'
CAPTURE = P / 'fit-000/raw-joint-capture-563/raw-direction-001'
BASIS = CAPTURE / 'basis-and-delta.npz'
RAW = CAPTURE / 'candidate.npz'
EMITTED = P / 'fit-000/candidate.npz'

PINS = {
 'base_candidate': '58488b5d329c3506134c290c6a3435518e1ad75878ac451fe2a0d3179996bc48',
 'sat_pairs': '724a77943714b4ffc53ab1819138d2a4ebade87c1083c8a053d30234bc0f51c5',
 'target_npz': '5f1033ab73ae09bdbdd2046ef63ec939b663d0ac87ab5aa2f27007e7e92c4e04',
 'target_report': 'aef02f6c2ab8e29ee65b4556ebc15f004e8360014ea73c491c1f3a41ab1885e4',
 'fit_report': 'bd8f0c62b8ec11418ec0e003b2c99df133e9513258925ce31f99afe9693d90d9',
 'fit_candidate': 'd8df80623f6a9bc34968ed62efa7d84ff9905563dd4cab027e1ec8480d86d59a',
 'raw_basis': 'f9102af899a2c938482697ffcaebb2bdaddeca3446fc129eb896645d5d4a8fef',
 'raw_candidate': 'fd9337a74a692b33feb101650317c8937dd182e1e511d6e8b107ebbe4288437d',
 'audit159': 'c531b93f4653c0d2694b205ccc0d56385dc22ad95a3c75db2d5755ae9a94bd25',
 '458_owner': '056c8e60dad54f92a9c3ab69498dbfac0dc2c0152be272eadea2a58a3b6ea0e0',
}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_npz(path): return np.load(path, allow_pickle=False)
def allclose_exact(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def import_owner(path):
    spec=importlib.util.spec_from_file_location('owner458_633',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def soft_metrics(C,h,w,meta,delta):
    v=np.asarray(C @ np.asarray(delta,dtype=np.float64).reshape(-1)).reshape(-1)
    frac=v/h
    unmet=np.maximum(0.0,1.0-frac)
    pair_rows={}
    for i,m in enumerate(meta): pair_rows.setdefault(int(m['pair_index']),[]).append(i)
    met=np.array([np.all(v[ix] >= h[ix]-1e-12) for ix in pair_rows.values()],dtype=bool)
    return {'rows':len(h),'pairs_with_positive_rows':len(pair_rows),
      'satisfied_rows':int(np.count_nonzero(v >= h-1e-12)),
      'satisfied_pairs_all_positive_rows':int(np.count_nonzero(met)),
      'weighted_unmet_fraction':float(np.dot(w,unmet)),
      'minimum_progress_mm':float(v.min(initial=np.inf)),
      'minimum_progress_fraction':float(frac.min(initial=np.inf)),
      'minimum_target_slack_mm':float(np.min(v-h,initial=np.inf)),
      'violated_rows':int(np.count_nonzero(v < h-1e-12))}

def main():
    paths={'base_candidate':BASE,'audit159':AUDIT,'458_owner':OWNER,'fit_report':FIT,
      'target_report':TARGET_REPORT,'target_npz':TARGETS,'sat_pairs':SAT,
      'raw_basis':BASIS,'raw_candidate':RAW,'fit_candidate':EMITTED}
    absent=[str(p) for p in paths.values() if not p.is_file()]
    if absent: raise SystemExit('missing pinned input(s): '+', '.join(absent))
    actual={k:sha(v) for k,v in paths.items()}
    source_603=ROOT/'D/surface-clearance-546/component-scp-area-backtrack-603/frozen-v1/component_scp_backtracking603.py'
    source_408=ROOT/'D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
    source_466=ROOT/'D/D/triangle-sat-rows-466/triangle_sat_rows_466.py'
    source_hashes={str(x):sha(x) for x in (source_603,source_408,source_466)}
    source_expected={str(source_603):'f02aed5e462779e85a5d4df46a38a68bcf7eb3972c9dbf565830c4919b387be6',
      str(source_408):'5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2',
      str(source_466):'cf4df8cd8a3f2773f0992a5f9a1d9b899d64db781b0a58cc2ed618285fa11cf3'}
    if source_hashes!=source_expected: raise SystemExit('source owner pin mismatch')
    for k,v in PINS.items():
      if actual.get(k)!=v: raise SystemExit(f'pin mismatch {k}: {actual.get(k)} != {v}')
    base=read_npz(BASE); targets=read_npz(TARGETS); sat=read_npz(SAT); bz=read_npz(BASIS)
    fit=json.loads(FIT.read_text()); trep=json.loads(TARGET_REPORT.read_text()); audit=json.loads(AUDIT.read_text())
    if not trep.get('matched_159_complete') or int(trep.get('crossing_pairs',-1))!=len(targets['pair_index']):
      raise SystemExit('target report not complete/matched')
    ident=np.asarray(sat['pair_identity_body_bone_face_triangle'],dtype=np.int64)
    want=np.column_stack((targets['bank_body'],targets['bank_stable_bone_id'],targets['face_rows'],targets['bank_triangle_stable_id'])).astype(np.int64)
    if not np.array_equal(ident,want): raise SystemExit('SAT/target pair identities do not match rowwise')
    if len(ident)!=len(targets['pair_index']): raise SystemExit('SAT/target row count mismatch')
    if not allclose_exact(bz['base_points_world_m_f32'],base['points_world_m']): raise SystemExit('raw basis is bound to another base point array')
    scope=np.asarray(bz['output_scope_rows'],dtype=np.int64)
    anchors=np.asarray(bz['anchor_rows'],dtype=np.int64)
    delta=np.asarray(bz['delta_mm'],dtype=np.float64).reshape(-1)
    if len(scope)*3!=len(delta) or len(np.unique(scope))!=len(scope): raise SystemExit('basis delta/scope mismatch')
    points_mm=np.asarray(base['points_world_m'],dtype=np.float64)*1000.0
    faces=np.asarray(base['source_O_faces'],dtype=np.int64)
    pairs=[]; bone_map={}
    for i,(body,bone,skin_face,tri) in enumerate(ident):
      pairs.append({'face_row':int(sat['skin_face_rows'][i]),'body':int(body),'bone_id':int(bone),'triangle_id':int(tri)})
      bone_map[(int(body),int(bone),int(tri))]=np.asarray(sat['fixed_triangles_world_mm'][i],dtype=np.float64)
    owner=import_owner(OWNER)
    built=owner.build_fixed_bone_outward_plane_rows(points_mm,faces,scope,anchors,pairs,bone_map,total_weight=1.0)
    C,h,w,meta=built['C'],built['h'],built['weights'],built['metadata']
    c_report=fit['constraints']
    if len(h)!=int(c_report['clearance_linear_rows']) or not np.isclose(w.sum(),1.0,atol=1e-12):
      raise SystemExit(f"objective row mismatch built={len(h)} report={c_report['clearance_linear_rows']} weights={w.sum()}")
    # Current emitted fitter candidate and capture candidate are full-coordinate F32.
    rawz=read_npz(RAW); emitted=read_npz(EMITTED)
    raw_f32=np.asarray(rawz['points_world_m'],dtype=np.float32)
    fit_f32=np.asarray(emitted['points_world_m'],dtype=np.float32)
    base_f32=np.asarray(base['points_world_m'],dtype=np.float32)
    if raw_f32.shape!=base_f32.shape or fit_f32.shape!=base_f32.shape: raise SystemExit('point domain mismatch')
    def d_from_points(pt): return ((pt[scope].astype(np.float64)-base_f32[scope].astype(np.float64))*1000.0).reshape(-1)
    raw_emitted=soft_metrics(C,h,w,meta,d_from_points(raw_f32))
    fit_emitted=soft_metrics(C,h,w,meta,d_from_points(fit_f32))
    raw_lp=soft_metrics(C,h,w,meta,delta)
    # Same order as frozen 408.emit_raw_lp_points: (x_mm + d)/1000, then F32.
    uniform={}
    for alpha in (0.5,0.25,0.0625):
      pt=base_f32.copy()
      pt[scope]=((points_mm[scope]+alpha*delta.reshape(-1,3))/1000.0).astype(np.float32)
      uniform[str(alpha)]=soft_metrics(C,h,w,meta,d_from_points(pt))
    iters=fit.get('area_scp_iterations',[])
    component=[]
    for it in iters:
      bk=it.get('exact_f32_area_backtrack') or {}; cw=bk.get('component_scp_backtracking603') or {}
      events=cw.get('components',[])
      if events:
        counts={}
        failed={}
        for ev in events:
          a=ev.get('selected_alpha')
          key=str(a) if a is not None else 'none'
          counts[key]=counts.get(key,0)+1
          if not ev.get('accepted'): failed[ev.get('reason','unknown')]=failed.get(ev.get('reason','unknown'),0)+1
        component.append({'iteration':it.get('iteration'),'components':len(events),'selected_alpha_counts':counts,'rejected_reason_counts':failed,
          'example_trials':[{k:v for k,v in tr.items() if k in ('alpha','accepted','reason','edge_gradient_pass','minimum_clearance_progress_mm')} for ev in events[:1] for tr in ev.get('trials',[])]})
    report={
      'schema':'numi.human.clearance_objective_audit_633.v1',
      'classification':'arithmetic-only diagnosis; no fit, geometry audit, or admission',
      'inputs':{k:{'path':str(v),'sha256':actual[k]} for k,v in paths.items()},
      'owners':{'603_component_scp_path':str(source_603),'603_component_scp_sha256':source_hashes[str(source_603)],
        '408_gate_path':str(source_408),'408_gate_sha256':source_hashes[str(source_408)],
        '466_triangle_sat_path':str(source_466),'466_triangle_sat_sha256':source_hashes[str(source_466)]},
      'analysis_helper':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).resolve()),
        'replay_command':'/Users/n/numi-human-prep-venv-20261005/bin/python '+str(Path(__file__).resolve())},
      'identity_check':{'target_rows':len(want),'sat_rows':len(ident),'rowwise_body_bone_skinface_triangle_equal':True,'audit159_path':str(AUDIT),'audit159_sha256':actual['audit159'],'audit159_status':audit.get('status')},
      'objective_rebuild':{'rows':len(h),'pairs_with_positive_targets':len(set(int(m['pair_index']) for m in meta)),
        'weights_sum':float(w.sum()),'report_weighted_unmet':c_report['clearance_progress']['weighted_unmet_fraction'],
        'raw_lp_delta':raw_lp,'raw_capture_F32':raw_emitted,'fit_emitted_F32':fit_emitted,
        'uniform_alpha_emitted_F32_from_raw_delta':uniform},
      'fit_construction_gates':{'status':fit.get('status'),'source_cap_mm':fit.get('source_cap_mm'),
        'maximum_source_displacement_mm':fit.get('maximum_source_displacement_mm'),
        'maximum_emitted_node_step_mm':fit.get('maximum_emitted_node_step_mm'),
        'emitted_f32_checks':{k:fit['emitted_f32_checks'].get(k) for k in ('alpha','anchors_bitwise_unchanged','edge_gradient_emission_pass','edge_gradient_minimum_emitted_slack_mm','maximum_node_step_mm','maximum_source_displacement_mm','strict_f32_emission_gate')},
        'area_gate':fit.get('global_f32_area_gate')},
      'component_alpha_receipts':component,
      '603_backtracking_assessment':{
        'source_location':'component_scp_backtracking603.py:96-128; original 408 backtrack at propose_joint_clearance_witnesses_408.py:734-905',
        'observed_behavior':'603 checks A/b for current and raw LP deltas, audits A/C/edge row ownership, then passes each component target to original 408 backtracking. 408 tests strict F32 area, source cap, trust, anchors, and edge gradients; it does not receive a separately appended positive face C/h endpoint family. After component composition, 603 records A@delta-b minimum as a diagnostic and computes soft C/h slack but does not reject on either diagnostic.',
        'hard_row_implication':'If positive endpoint C/h rows are appended to A/b, the unchanged baseline is infeasible and 603 current-delta preflight will reject/fallback. If rows are kept outside A/b, 603 can accept an F32 output that violates them unless an explicit endpoint certificate is added.',
        'required_fail_closed_check':'Keep original A/b baseline checks. Treat new positive C/h as a separate endpoint contract. Require raw LP C@delta>=h; after exact 408 F32 emission check row-owned C@delta_f32>=h with no arbitrary tolerance; after composition require every row. Do not apply endpoint h to the unchanged baseline. Preserve all 408 gates and the existing complete 545/185 checks.',
        'current_emitted_violation_count':len(h)-fit_emitted['satisfied_rows'],
        'uniform_alpha_emitted_violation_counts':{a:len(h)-m['satisfied_rows'] for a,m in uniform.items()},
        'no_dual_or_iis':'No IIS or dual data is present in the retained report.'},
      'interpretation':{
        'soft_vs_hard':'458 C/h rows are soft normalized-slack objectives in this run. They are not clearance certificates; default 408/603 area/source/trust/anchor/edge gates do not enforce them.',
        'measured_emission':'The accepted F32 endpoint passes the recorded area/source/trust/anchor/edge gates, yet its rebuilt 458 positive rows remain unsatisfied; therefore 603 can emit a point violating added positive C/h endpoint rows unless they are separately checked.',
        'alphas':'Uniform alpha rows are arithmetic-only demonstrations from the raw direction. 603 component alpha is nonuniform and each component trial calls frozen 408 backtracking; the component receipts are preserved separately.',
        'recommended_certificate':'Keep original 408 A/b gate rows unchanged and distinguish any new C/h endpoint rows. Verify raw LP endpoint C/h and row-owner partition. For each component, check its F32-emitted rows against their endpoint rhs inside the alpha trial before accepting; after composition require all emitted F32 C/h rows pass. Do not demand the positive endpoint rows of the unchanged baseline, and do not treat a damped component as certified if it misses its rhs. If a full endpoint cannot pass all C/h, retain the failed witness and do not claim hard-face clearance. Full 545 plus complete185/self audits remain mandatory.',
        'limitation':'The component alpha receipt does not contain every trial coordinate array; alpha-scaled raw-field metrics are not a reconstruction of nonuniform componentwise emissions.'}}
    out=Path(sys.argv[1]) if len(sys.argv)>1 else (ROOT/'D/clearance-objective-audit-633/objective-audit-633.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(out),'sha256':sha(out),'rows':len(h),'raw_lp':raw_lp,'raw_f32':raw_emitted,'fit_f32':fit_emitted,'uniform_alpha':uniform,'component_alpha':component},sort_keys=True))
if __name__=='__main__': main()
