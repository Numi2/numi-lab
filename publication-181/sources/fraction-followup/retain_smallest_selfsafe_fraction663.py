#!/usr/bin/env python3
"""Retain the smallest screened self-safe step with best pair reduction per component."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

SELECT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-merit-retention-631/select_components631_v2.py')
SELECT_SHA='31e9dc21071aed2f27399c2ef35da6133d4444e7a300870ffa5c9c541b3e269b'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('refusing overwrite')
    s=json.loads(a.screen.read_text())
    if s.get('status')!='CONSTRUCTION_SCREEN_PROPOSAL' or s.get('trial_full159_parity') is not True:raise ValueError('invalid screen')
    if sha(SELECT)!=SELECT_SHA:raise ValueError('component selector changed')
    for kind in ('baseline','trial'):
        if sha(s[kind])!=s[kind+'_sha256']:raise ValueError('changed screen input '+kind)
    if sha(s['baseline_report159'])!=s['baseline_report159_sha256'] or sha(s['trial_report159'])!=s['trial_report159_sha256']:raise ValueError('changed 159 report')
    spec=importlib.util.spec_from_file_location('component631retain',SELECT);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    b=load(s['baseline']);t=load(s['trial']);x=b['points_world_m'];y=t['points_world_m'];faces=b['source_O_faces']
    if not np.array_equal(faces,t['source_O_faces']):raise ValueError('topology changed')
    labels,_=m.labels_for_changes(x,y,faces)
    eligible=[r for r in s['fraction_results'] if r.get('no_bank_count_increase') is True and r.get('bone_target_improvement') is True and r.get('exact_incremental_self_pair_count')==0]
    chosen=[]
    for c in sorted({r['component'] for r in eligible}):
        choices=[r for r in eligible if r['component']==c]
        chosen.append(min(choices,key=lambda r:(r['surface_total'],sum(r['target_deltas'].values()),r['fraction'])))
    if not chosen:raise ValueError('no eligible component')
    out={k:v.copy() for k,v in b.items()};delta=y.astype(np.float64)-x.astype(np.float64)
    for r in chosen:
        nodes=np.flatnonzero(labels==r['component'])
        if len(nodes)!=r['node_count']:raise ValueError('component inventory changed')
        out['points_world_m'][nodes]=(x[nodes].astype(np.float64)+r['fraction']*delta[nodes]).astype(np.float32)
    anchors=b['typed_coarse_anchor_output_rows_018']
    if not np.array_equal(x[anchors],out['points_world_m'][anchors]):raise ValueError('anchor changed')
    if any(not m.same(b[k],out[k]) for k in b if k!='points_world_m'):raise ValueError('nonpoint changed')
    a.output.mkdir(parents=True)
    cp=a.output/'candidate.npz';np.savez_compressed(cp,**out)
    disp=np.linalg.norm((out['points_world_m'].astype(np.float64)-x.astype(np.float64))*1000,axis=1)
    receipt={'status':'SELF_SCREENED_PROPOSAL_REQUIRES_FULL_545','screen':str(a.screen),'screen_sha256':sha(a.screen),
      'baseline_sha256':s['baseline_sha256'],'trial_sha256':s['trial_sha256'],'component_selector_sha256':SELECT_SHA,
      'selected':chosen,'candidate':str(cp),'candidate_sha256':sha(cp),'changed_nodes':int(np.count_nonzero(disp)),
      'maximum_step_mm':float(disp.max(initial=0.)),'anchors_and_nonpoint_arrays_unchanged':True,
      'native_admission':False,'accepted_physical_steps':0}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
