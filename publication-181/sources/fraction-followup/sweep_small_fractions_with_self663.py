#!/usr/bin/env python3
"""Screen connected partial steps against the pinned exact 159 bone-pair owner.

Construction-only: exact self, containment, quality and physical gates remain mandatory.
The full trial audit must first match the incremental changed-face reconstruction.
"""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546')
TOPO=N/'topology/topology-bone-aware-025.py'
SELECT=N/'component-merit-retention-631/select_components631_v2.py'
PAIRS=((43,16),(65,99),(109,114))

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-state',type=Path,required=True)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--trial-report159',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-seconds',type=int,default=120)
    a=p.parse_args()
    if a.output.exists():raise ValueError('refusing overwrite')
    st=json.loads(a.baseline_state.read_text())
    for k in ('candidate','point_summary','surface_report'):
        if sha(st[k])!=st[k+'_sha256']:raise ValueError('baseline state hash mismatch '+k)
    base=load(st['candidate']);trial=load(a.trial)
    x=base['points_world_m'];y=trial['points_world_m'];faces=base['source_O_faces']
    if not np.array_equal(faces,trial['source_O_faces']):raise ValueError('trial topology differs')
    if not np.array_equal(x[base['typed_coarse_anchor_output_rows_018']],y[base['typed_coarse_anchor_output_rows_018']]):
        raise ValueError('trial moves typed anchors')
    selector=mod('component631sweep',SELECT)
    prior=selector.bank_pairs(json.loads(Path(st['surface_report']).read_text()),st['candidate_sha256'],len(faces))
    expected_trial=selector.bank_pairs(json.loads(a.trial_report159.read_text()),sha(a.trial),len(faces),set(prior))
    topo=mod('topo025sweep',TOPO)
    if sha(topo.OWNER370)!=topo.EXPECTED['owner370']:raise ValueError('counter owner changed')
    owner=mod('counter370sweep',topo.OWNER370)
    _,surf,bank_helper,exact,runtime=owner.load_pinned_runtime()
    raw=bank_helper.read_bank_inputs(owner.BANK_DIR_DEFAULT,owner.EVIDENCE_ROOT_DEFAULT)
    banks,keys=owner.typed_banks_from_159_inputs(raw)
    if set(keys)!=set(prior):raise ValueError('typed bank inventory mismatch')
    labels,face_owner=selector.labels_for_changes(x,y,faces)
    changed=np.flatnonzero(face_owner>=0)
    start=time.monotonic();a.output.mkdir(parents=True)
    pred_path=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py')
    if sha(pred_path)!='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d':raise ValueError('self-predicate source changed')
    sys.path.insert(0,str(pred_path.parent.parent))
    pred=mod('numilab_human._fraction_sweep663',pred_path)
    base_records,_=pred.float32_surface_records(x,faces,source_seam_groups=())
    self_base,_=pred._prepare_complete_self_pair_baseline(base_records)
    if self_base.triangle_pairs:raise ValueError('baseline self pairs are nonzero')
    def progress(e):
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(e,sort_keys=True)+'\n')
        print(json.dumps(e,sort_keys=True),flush=True)
    def recalculated(points,rows):
        counter=owner.Exact159PatchCounter(points,banks,exact,surf.records)
        new=counter.count_faces(rows,faces[rows])
        if set(new)!=set(prior):raise ValueError('counter omitted bank')
        affected=set(map(int,rows))
        return {k:{p for p in prior[k] if p[0] not in affected}|set(map(tuple,new[k]['triangle_pairs'])) for k in keys}
    full=recalculated(y,changed)
    if any(full[k]!=expected_trial[k] for k in keys):raise ValueError('changed-face reconstruction does not match complete trial audit')
    progress({'stage':'exact_trial_parity','changed_face_count':len(changed),'elapsed_s':time.monotonic()-start})
    comps=sorted({int(face_owner[f]) for k in PAIRS for f,_ in prior[k] if face_owner[f]>=0})
    results=[];best={};delta=y.astype(np.float64)-x.astype(np.float64)
    fractions=tuple(i/128. for i in range(1,17))
    for c in comps:
        nodes=np.flatnonzero(labels==c);rows=np.flatnonzero(face_owner==c)
        baseline_total=sum(map(len,prior.values()));best_row=None
        for fraction in fractions:
            if time.monotonic()-start>a.deadline_seconds:
                progress({'stage':'deadline','component':c,'elapsed_s':time.monotonic()-start});break
            v=x.copy();v[nodes]=(x[nodes].astype(np.float64)+fraction*delta[nodes]).astype(np.float32)
            after=recalculated(v,rows)
            differences={str(k[0])+'/'+str(k[1]):len(after[k])-len(prior[k]) for k in keys if len(after[k])!=len(prior[k])}
            target_changes={str(k[0])+'/'+str(k[1]):len(after[k])-len(prior[k]) for k in PAIRS}
            safe_count=all(len(after[k])<=len(prior[k]) for k in keys)
            target_delta=sum(target_changes.values());total=sum(map(len,after.values()))
            r={'component':c,'fraction':fraction,'node_count':len(nodes),'affected_faces':len(rows),'surface_total':total,
               'total_delta':total-baseline_total,'target_deltas':target_changes,'all_bank_deltas':differences,
               'no_bank_count_increase':safe_count,'bone_target_improvement':target_delta<0}
            if safe_count and target_delta<0:
                rec,_=pred.float32_surface_records(v,faces,source_seam_groups=())
                _,self_report=pred._update_complete_self_pair_baseline(self_base,rec)
                r['exact_incremental_self_pair_count']=int(self_report['count'])
                r['exact_incremental_self_pair_ids']=self_report['triangle_pairs']
            else:
                r['exact_incremental_self_pair_count']=None
            results.append(r);progress({'stage':'fraction',**r,'elapsed_s':time.monotonic()-start})
            if safe_count and target_delta<0 and r['exact_incremental_self_pair_count']==0 and (best_row is None or (total,target_delta,-fraction)<(best_row['surface_total'],sum(best_row['target_deltas'].values()),-best_row['fraction'])):
                best_row=r
        if best_row is not None:best[c]=best_row
        if time.monotonic()-start>a.deadline_seconds:break
    out_candidate=None
    if best:
        out={k:v.copy() for k,v in base.items()}
        for c,r in best.items():
            nodes=np.flatnonzero(labels==c);out['points_world_m'][nodes]=(x[nodes].astype(np.float64)+r['fraction']*delta[nodes]).astype(np.float32)
        out_candidate=a.output/'candidate.npz';np.savez_compressed(out_candidate,**out)
    summary={'status':'CONSTRUCTION_SCREEN_PROPOSAL' if best else 'NO_TARGET_IMPROVEMENT_IN_SCREENED_FRACTIONS',
      'baseline':st['candidate'],'baseline_sha256':st['candidate_sha256'],'baseline_report159':st['surface_report'],'baseline_report159_sha256':st['surface_report_sha256'],
      'trial':str(a.trial),'trial_sha256':sha(a.trial),'trial_report159':str(a.trial_report159),'trial_report159_sha256':sha(a.trial_report159),
      'counter_owner':str(topo.OWNER370),'counter_owner_sha256':sha(topo.OWNER370),'runtime_pins':runtime,'trial_full159_parity':True,
      'target_components':comps,'fraction_results':results,'selected_fractions':list(best.values()),
      'output_candidate':str(out_candidate) if out_candidate else None,'output_candidate_sha256':sha(out_candidate) if out_candidate else None,
      'deadline_seconds':a.deadline_seconds,'seconds':time.monotonic()-start,
      'native_admission':False,'accepted_physical_steps':0,
      'limitation':'Exact 159 changed-face and incremental self screening only. Full 130/159, containment, area, mass, mechanical transfer and native qualification remain mandatory.'}
    (a.output/'report.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':summary['status'],'target_components':comps,'selected':summary['selected_fractions'],'seconds':summary['seconds']}),flush=True)
if __name__=='__main__':main()
