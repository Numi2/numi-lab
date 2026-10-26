#!/usr/bin/env python3
"""Single-flip fixed-coordinate area conditioning for the round-008 limiter.

Uses the pinned 402 geometric flip guard, 405 CSR replay, and 541 quality
metrics. Emits at most one candidate. It does not claim exact self or bone
nonincrease; run the unchanged 130/159 gates before reusing the topology.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
D=ROOT/'D'
DEFAULT_BASE=D/'surface-clearance-546/topology/round-007-wide-chain-025/round-004/candidate.npz'
DEFAULT_SELF_SUMMARY=D/'surface-clearance-546/topology/round-007-wide-chain-025/final-check/audit-130/summary.json'
DEFAULT_SELF_PAIRS=D/'surface-clearance-546/topology/round-007-wide-chain-025/final-check/audit-130/surface-self-pairs.npy'
DEFAULT_SURFACE=D/'surface-clearance-546/topology/round-007-wide-chain-025/final-check/audit-159/report.json'
DEFAULT_AREA_FIT=D/'surface-clearance-546/round-008-barycentric/fit/report.json'
OWNER402=D/'local-two-parent-retriangulation-402/two_parent_patch_flips_402.py'
REPLAY405=D/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
QUALITY541=D/'focused-clearance-541/topology/repair_focused_clearance_541.py'
BASE_SHA='eadcd3838ec5bf4f2d72d45ba5d19c775e9fef4361f92ebea6f49e18d4bfe3a5'
SELF_SUMMARY_SHA='9432674df45bc47eb0d968bf30e115dec7d2f35ba9d03b072f22ad68b91728c4'
SELF_PAIRS_SHA='55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de'
SURFACE_SHA='ddf1211aa7ca01ef8f1c907c89eb082ab700f00299db09b74fa6426bafc270f1'
AREA_FIT_SHA='9dacea662878b9fc892e7eb80656eee661b623eda9ff8408d897dc3f4893ab90'
OWNER402_SHA='f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764'
REPLAY405_SHA='e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967'
QUALITY541_SHA='26930be05518834f02724948cffa5ff74719f070a4ac4e4917b0965f42a412aa'
AREA_FACE=22027
AREA_FLOOR=0.10
ANGLE_FLOOR_DEG=2.0
MAX_LOCAL_MIN_ANGLE_LOSS_DEG=0.75

class ProposalError(ValueError): pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def identical(a,b):
    x,y=np.ascontiguousarray(a),np.ascontiguousarray(b)
    return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ProposalError(f'cannot import pinned owner: {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def verify_pins(base_path,self_summary_path,self_pairs_path,surface_path,area_fit_path):
    expected=((OWNER402,OWNER402_SHA,'402 flip owner'),(REPLAY405,REPLAY405_SHA,'405 ancestry replay'),
              (QUALITY541,QUALITY541_SHA,'541 quality owner'),(base_path,BASE_SHA,'baseline candidate'),
              (self_summary_path,SELF_SUMMARY_SHA,'baseline self summary'),(self_pairs_path,SELF_PAIRS_SHA,'baseline self-pair file'),(surface_path,SURFACE_SHA,'baseline 159 report'),
              (area_fit_path,AREA_FIT_SHA,'round-008 area fit report'))
    out={}
    for p,want,label in expected:
        if not Path(p).is_file(): raise ProposalError(f'{label} missing: {p}')
        got=sha(p)
        if got!=want: raise ProposalError(f'{label} hash mismatch: {got} != {want}')
        out[label]={'path':str(p),'sha256':got}
    return out

def parse_witness_rows(area_report):
    rows={}
    for it in area_report.get('area_scp_iterations',[]):
        back=it.get('exact_f32_area_backtrack') or {}
        for trial in back.get('trials',[]):
            gate=trial.get('area_gate') or {}
            bad=gate.get('bad_fixed_037_reference_face_rows_first_64') or []
            if gate.get('bad_fixed_037_reference_face_count',0):
                for row in bad:
                    row=int(row);rows[row]=rows.get(row,0)+1
    if AREA_FACE not in rows: raise ProposalError(f'pinned limiting face {AREA_FACE} is absent from backtrack witnesses')
    return sorted(rows,key=lambda r:(-rows[r],r)),rows

def parse_current_orientation_witness_rows(area_report):
    rows={}
    for it in area_report.get('area_scp_iterations',[]):
        back=it.get('exact_f32_area_backtrack') or {}
        for trial in back.get('trials',[]):
            gate=trial.get('area_gate') or {}
            if gate.get('bad_current_oriented_face_count',0):
                for raw in gate.get('bad_current_oriented_face_rows_first_64') or []:
                    row=int(raw);rows[row]=rows.get(row,0)+1
    return sorted(rows,key=lambda r:(-rows[r],r)),rows

def check_exact_nonincrease(before_report,after_report):
    """Return per-bank counts only for two complete candidate-bound 185-bank reports."""
    def extract(doc):
        if (doc.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT' or
            doc.get('complete') is not True or doc.get('expected_bank_count')!=185 or doc.get('bank_count')!=185):
            raise ProposalError('all-185 exact report is incomplete')
        counts={}
        for b in doc.get('banks',[]):
            k=(int(b['body']),int(b['stable_bone_id']))
            if k in counts: raise ProposalError(f'duplicate bank in exact report: {k}')
            counts[k]=len((b.get('result') or {}).get('triangle_pairs') or [])
        if len(counts)!=185: raise ProposalError(f'exact report has {len(counts)} banks, expected 185')
        return counts
    before,after=extract(before_report),extract(after_report)
    if before.keys()!=after.keys(): raise ProposalError('typed bank identities changed between exact reports')
    increases={f'{b}/{s}':(before[(b,s)],after[(b,s)]) for b,s in before if after[(b,s)]>before[(b,s)]}
    return {'passed':not increases,'before_total':sum(before.values()),'after_total':sum(after.values()),
            'increased_banks':increases,'checked_bank_count':185}

def quality_gate(before,after,face_rows,*,area_floor=AREA_FLOOR,angle_floor=ANGLE_FLOOR_DEG,max_angle_loss=MAX_LOCAL_MIN_ANGLE_LOSS_DEG):
    ix=np.asarray(sorted(set(map(int,face_rows))),dtype=np.int64)
    if ix.size!=2: raise ProposalError('a flip must replace exactly two faces')
    old_area=float(np.min(before['area_ratio'][ix]));new_area=float(np.min(after['area_ratio'][ix]))
    old_angle=float(np.min(before['min_angle_deg'][ix]));new_angle=float(np.min(after['min_angle_deg'][ix]))
    old_cos=float(np.min(before['normal_cos'][ix]));new_cos=float(np.min(after['normal_cos'][ix]))
    global_old_area=float(np.min(before['area_ratio']));global_new_area=float(np.min(after['area_ratio']))
    global_old_angle=float(np.min(before['min_angle_deg']));global_new_angle=float(np.min(after['min_angle_deg']))
    global_old_cos=float(np.min(before['normal_cos']));global_new_cos=float(np.min(after['normal_cos']))
    passed=(new_area>=area_floor and new_cos>0.0 and new_angle>=angle_floor and
            new_area>old_area+1e-12 and new_angle-old_angle>=-max_angle_loss-1e-10 and
            global_new_area>=global_old_area-1e-12 and global_new_angle>=global_old_angle-1e-10 and global_new_cos>=global_old_cos-1e-10)
    return {'passed':bool(passed),'area_floor':area_floor,'angle_floor_deg':angle_floor,
            'local_old_min_area_ratio':old_area,'local_new_min_area_ratio':new_area,
            'global_old_min_area_ratio':global_old_area,'global_new_min_area_ratio':global_new_area,
            'local_old_min_angle_deg':old_angle,'local_new_min_angle_deg':new_angle,
            'global_old_min_angle_deg':global_old_angle,'global_new_min_angle_deg':global_new_angle,
            'global_old_min_normal_cosine':global_old_cos,'global_new_min_normal_cosine':global_new_cos,
            'local_old_min_normal_cosine':old_cos,'local_new_min_normal_cosine':new_cos,
            'strict_local_worst_area_improvement':bool(new_area>old_area+1e-12),
            'global_worst_area_nonregression':bool(global_new_area>=global_old_area-1e-12),
            'local_angle_loss_within_bound':bool(new_angle-old_angle>=-max_angle_loss-1e-10),
            'absolute_angle_gate_preserved':bool(new_angle>=angle_floor and global_new_angle>=global_old_angle-1e-10 and global_new_cos>=global_old_cos-1e-10)}

def boundaries(faces, owner):
    em=owner.edge_map(faces)
    bad=[(e,v) for e,v in em.items() if len(v)>2]
    if bad: raise ProposalError('input/output has nonmanifold edge')
    return {e for e,v in em.items() if len(v)==1}

def build_one(base,area_report,owner,replay,quality,*,seed_limit=64):
    owner.validate_candidate(base)
    F=np.asarray(base['source_O_faces']); RF=np.asarray(base['area_reference_037_source_O_faces'])
    if not np.array_equal(F,RF): raise ProposalError('current topology differs from fixed-037 connectivity')
    area_rows,frequency=parse_witness_rows(area_report)
    current_orientation_rows,current_orientation_frequency=parse_current_orientation_witness_rows(area_report)
    if AREA_FACE not in area_rows[:seed_limit]: raise ProposalError('limiting face fell outside bounded seed prefix')
    before=quality.face_quality(np.asarray(base['points_world_m'],dtype=np.float64)*1000,
                                np.asarray(base['area_reference_037_points_world_m_f64'],dtype=np.float64)*1000,F,RF)
    if float(np.min(before['area_ratio']))<AREA_FLOOR:
        raise ProposalError('pinned baseline violates the existing whole-mesh fixed-037 area floor')
    bnd0=boundaries(F,owner)
    em=owner.edge_map(F)
    src=quality.csr_rows(base['face_source_ancestry_indptr'],base['face_source_ancestry_face_rows'],len(F))
    sub=quality.csr_rows(base['face_fine_subface_ancestry_indptr'],base['face_fine_subface_ancestry_rows'],len(F))
    detail=replay.detail_csr(base,set())
    candidate=[]; rejection={}
    seeds=area_rows[:seed_limit]
    for seed in seeds:
        for a,b in zip(F[seed],np.roll(F[seed],-1)):
            edge=(min(int(a),int(b)),max(int(a),int(b)))
            inc=em.get(edge,[])
            if len(inc)!=2: rejection['not_two_face_edge']=rejection.get('not_two_face_edge',0)+1;continue
            rows=sorted(int(v[0]) for v in inc)
            if seed not in rows: raise ProposalError('seed incidence mismatch')
            if not quality.compatible(base,*rows,src,sub,detail): rejection['source_incompatible']=rejection.get('source_incompatible',0)+1;continue
            d=int(base['parent_fine_detail_face_row'][rows[0]])
            if d<0: rejection['non_single_detail']=rejection.get('non_single_detail',0)+1;continue
            op,why=owner.candidate_flip(base,edge,inc,{d},set(),-MAX_LOCAL_MIN_ANGLE_LOSS_DEG,0.0,em)
            if op is None: rejection[why]=rejection.get(why,0)+1;continue
            # Restrict to one operation touching a witnessed area limiter.
            trial=replay.apply_op(base,op)
            after=quality.face_quality(np.asarray(trial['points_world_m'],dtype=np.float64)*1000,
                                       np.asarray(trial['area_reference_037_points_world_m_f64'],dtype=np.float64)*1000,
                                       trial['source_O_faces'],trial['area_reference_037_source_O_faces'])
            gate=quality_gate(before,after,rows)
            if not gate['passed']:
                rejection['quality_gate_no_worst_area_gain_or_angle_fail']=rejection.get('quality_gate_no_worst_area_gain_or_angle_fail',0)+1;continue
            if boundaries(trial['source_O_faces'],owner)!=bnd0: raise ProposalError('flip changed mesh boundary')
            if not identical(base['points_world_m'],trial['points_world_m']): raise ProposalError('flip changed coordinate array')
            if not np.array_equal(trial['source_O_faces'],trial['area_reference_037_source_O_faces']): raise ProposalError('flip desynchronized fixed-037 faces')
            anchors=np.asarray(base['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
            if anchors.size!=237 or not identical(base['points_world_m'][anchors],trial['points_world_m'][anchors]): raise ProposalError('typed anchors changed')
            changed={k for k in base if not identical(base[k],trial[k])}
            if changed-set(quality.ALLOWED_CHANGED_KEYS): raise ProposalError(f'unapproved arrays changed: {sorted(changed-set(quality.ALLOWED_CHANGED_KEYS))}')
            candidate.append({'edge':edge,'rows':rows,'seed':int(seed),'op':op,'trial':trial,'quality':gate})
    if not candidate:
        return None,{'status':'NO_SOURCE_COMPATIBLE_QUALITY_FLIP','witness_rows':seeds,'witness_frequency':frequency,
                     'current_oriented_area_witness_rows':current_orientation_rows,
                     'current_oriented_area_witness_frequency':current_orientation_frequency,
                     'candidate_count':0,'rejection_counts':rejection,'all185_nonincrease':'NOT_CHECKED',
                     'exact_self_zero':'NOT_CHECKED','physical_or_native_admission':False}
    # Better global minimum fixed-037 area ratio first, then higher absolute minimum angle; deterministic edge tiebreak.
    candidate.sort(key=lambda x:(-x['quality']['global_new_min_area_ratio'],
                                 -x['quality']['global_new_min_angle_deg'],
                                 -x['quality']['local_new_min_angle_deg'],x['edge'],x['rows']))
    best=candidate[0]
    meta={'schema':'numi.human.area_quality_conditioner_548.v1','status':'QUALITY_CONDITIONED_TOPOLOGY_PROPOSAL_PENDING_EXACT_AUDITS',
          'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,
          'area_backtrack_witness_face_rows':seeds,'witness_frequency':frequency,
          'current_oriented_area_witness_face_rows':current_orientation_rows,
          'current_oriented_area_witness_frequency':current_orientation_frequency,
          'priority_face_row':AREA_FACE,'eligible_candidate_count':len(candidate),
          'selected_edge':list(best['edge']),'selected_face_rows':best['rows'],'selected_seed_face_row':best['seed'],
          'selected_operation':best['op'],'quality_gate':best['quality'],'rejection_counts':rejection,
          'coordinates_bitwise_unchanged':True,'typed_anchor_coordinates_bitwise_unchanged':True,
          'source_cap_unchanged_from_input':True,'fixed037_orientation_and_area_floors_unchanged':True,
          'fixed037_connectivity_synchronized':True,'boundary_unchanged':True,
          'all185_nonincrease':'REQUIRES_COMPLETE_POSTBUILD_159_COMPARISON',
          'exact_self_zero':'REQUIRES_POSTBUILD_130_ZERO_PAIR_AUDIT',
          'lineage':'405 replay merges source/subface/immediate-parent CSR unions on the two replaced faces',
          'source_owner_hashes':{'402':OWNER402_SHA,'405':REPLAY405_SHA,'541':QUALITY541_SHA},
          'candidate_proposals_considered':len(candidate)}
    return best['trial'],meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',default=str(DEFAULT_BASE));ap.add_argument('--self-summary',default=str(DEFAULT_SELF_SUMMARY))
    ap.add_argument('--self-pairs',default=str(DEFAULT_SELF_PAIRS));ap.add_argument('--surface-report',default=str(DEFAULT_SURFACE))
    ap.add_argument('--area-fit-report',default=str(DEFAULT_AREA_FIT));ap.add_argument('--output-dir');ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args(); outdir=Path(a.output_dir) if a.output_dir else None
    pins=verify_pins(a.candidate,a.self_summary,a.self_pairs,a.surface_report,a.area_fit_report)
    with np.load(a.candidate,allow_pickle=False) as z: base={k:z[k] for k in z.files}
    selfdoc=json.loads(Path(a.self_summary).read_text()); pairs=np.asarray(np.load(a.self_pairs,allow_pickle=False),dtype=np.int64).reshape((-1,2))
    if sha(a.self_summary)!=SELF_SUMMARY_SHA or selfdoc.get('candidate_sha256')!=BASE_SHA or len(pairs)!=0:
        raise ProposalError('baseline must have pinned candidate-bound exact zero-self audit')
    if sha(a.surface_report)!=SURFACE_SHA: raise ProposalError('baseline surface report hash mismatch')
    surfacedoc=json.loads(Path(a.surface_report).read_text())
    if surfacedoc.get('candidate_sha256')!=BASE_SHA or surfacedoc.get('complete') is not True or surfacedoc.get('expected_bank_count')!=185:
        raise ProposalError('baseline exact 159 report must be complete and candidate-bound')
    baseline_crossing_count=sum(len((b.get('result') or {}).get('triangle_pairs') or []) for b in surfacedoc.get('banks',[]))
    if baseline_crossing_count!=2409: raise ProposalError(f'pinned selected baseline expected 2409 crossing pairs, got {baseline_crossing_count}')
    areadoc=json.loads(Path(a.area_fit_report).read_text())
    if areadoc.get('area_reference_037_for_area_constraints_only',{}).get('sha256')!=BASE_SHA:
        raise ProposalError('area report does not use pinned selected 037 reference')
    owner=loadmod('flip402_area548',OWNER402);replay=loadmod('replay405_area548',REPLAY405);quality=loadmod('quality541_area548',QUALITY541)
    trial,meta=build_one(base,areadoc,owner,replay,quality)
    base_quality=quality.face_quality(np.asarray(base['points_world_m'],dtype=np.float64)*1000,
                                      np.asarray(base['area_reference_037_points_world_m_f64'],dtype=np.float64)*1000,
                                      base['source_O_faces'],base['area_reference_037_source_O_faces'])
    meta['pinned_baseline_global_minimums']={'fixed037_area_ratio':float(np.min(base_quality['area_ratio'])),
        'normal_cosine':float(np.min(base_quality['normal_cos'])),'minimum_angle_deg':float(np.min(base_quality['min_angle_deg']))}
    meta['existing_unmodified_baseline_defects_allowed']=['negative fixed037 normal cosine','minimum angle below 2 degrees'];
    meta.update({'source_pins':pins,'baseline_exact_surface_pair_count':sum(len((b.get('result') or {}).get('triangle_pairs') or []) for b in surfacedoc.get('banks',[])),
                 'candidate_input_sha256':sha(a.candidate),'surface_report_sha256':sha(a.surface_report),'area_fit_report_sha256':sha(a.area_fit_report),
                 'self_pairs_sha256':sha(a.self_pairs),'self_pair_count':int(len(pairs))})
    if trial is None:
        print(json.dumps(meta,indent=2,sort_keys=True));return 2
    if a.preflight_only:
        meta['status']='PINNED_BASELINE_PREFLIGHT_READY_NO_CANDIDATE_WRITTEN'
        meta['preflight_only']=True
        print(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False));return 0
    if not a.output_dir: raise ProposalError('--output-dir is required unless --preflight-only is set')
    if outdir.exists(): raise FileExistsError(outdir)
    outdir.mkdir(parents=True);cand=outdir/'candidate.npz';np.savez_compressed(cand,**trial)
    meta.update({'candidate_path':str(cand),'candidate_sha256':sha(cand)})
    rp=outdir/'proposal-report.json';rp.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
    meta['report_path']=str(rp);meta['report_sha256']=sha(rp)
    print(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False));return 0

if __name__=='__main__': raise SystemExit(main())
