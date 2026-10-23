#!/usr/bin/env python3
"""Fail-closed ancestry audit for six frozen capitate attachment source IDs.

This script emits provenance evidence only. It does not choose or serialize new
attachment records. Run with the prepared Numi Python environment.
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

E = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT = E/'capitate-attachment-correspondence-001/source-slot-provenance-002'
BASE = E/'replacement-transfer-ancestry-082-parent-inference-001/wholebody-header-emitter-shadow-001/baseline/NumiHumanWholeBodySkinReduction.hpp'
CORR = E/'capitate-attachment-correspondence-001/capitate-attachment-correspondence.json'
REG = E/'native-pose-checks/connected-hand-registration069i/registered-skin.csv'
BINDINGS = E/'arm-skin-weight-audit.json'
G057 = E/'replacement-wrists-057/replacement-geometry.npz'
G069 = E/'finger-profile-069i-left-ring-web-retreat/replacement-geometry.npz'
A057 = E/'replacement-wrists-057'
A056 = E/'replacement-hands-056'
P049 = E/'replacement-hands-049'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def jload(path: Path):
    return json.loads(path.read_text())

corr=jload(CORR)
bindings={int(x['i']):x['name'] for x in jload(BINDINGS)['bindings']}
with REG.open(newline='') as f:
    reg={int(r['vertex']):r for r in csv.DictReader(f)}
z57=np.load(G057)
z69=np.load(G069)
map57={int(s):int(i) for i,s in zip(z57['internal_new_vertex_ids'],z57['emitted_source_slots'])}
map69={int(s):int(i) for i,s in zip(z69['internal_new_vertex_ids'],z69['emitted_source_slots'])}
assert map57==map69, '069i no longer has the exact 057 reallocation map; re-audit from current source'
assert len(map57)==len(z57['emitted_source_slots'])
anc057={side:jload(A057/f'{side}-source-ancestry.json') for side in ('left','right')}
anc056={side:jload(A056/f'{side}-source-ancestry.json') for side in ('left','right')}
anc049={side:jload(P049/f'{side}-source-ancestry.json') for side in ('left','right')}
patch049={side:np.load(P049/f'{side}-patch.npz')['source_slots'].astype(int) for side in ('left','right')}

sites=[]
for s in corr['sites']:
    slot=int(s['source_vertex'])
    assert slot in map57, f'frozen source slot {slot} was not found in emitted reallocation map'
    new_internal=map57[slot]
    a_side=next((side for side,d in anc057.items() if str(slot) in d),None)
    assert a_side is not None, f'no ancestry for emitted slot {slot}'
    a57=anc057[a_side][str(slot)]
    assert a57.get('side')==a_side
    expected_name=bindings[int(s['donor_body'])]
    expected_side='right' if expected_name.endswith('_r') else ('left' if expected_name.endswith('_l') else 'unknown')
    row=reg[slot]
    current_donor=int(row['donor'])
    current_name=bindings.get(current_donor,'unknown')
    chain={'stage057_emitted_slot':slot,'stage057_internal_new_vertex_id':new_internal,
           'stage057_ancestry':a57}
    ref056=a57.get('reference_candidate056_vertex')
    if ref056 is not None:
        a56=anc056[a_side].get(str(int(ref056)))
        chain['stage056_vertex']=int(ref056)
        chain['stage056_ancestry']=a56
        if a56 and a56.get('reference_candidate049_vertex') is not None:
            ref049=int(a56['reference_candidate049_vertex'])
            chain['stage049_vertex']=ref049
            idx=np.flatnonzero(patch049[a_side]==ref049)
            if len(idx)==1:
                chain['stage049_patch_local_index']=int(idx[0])
                chain['stage049_template_ancestry']=anc049[a_side][int(idx[0])]
            else:
                chain['stage049_patch_local_index_matches']=int(len(idx))
    nearest=s.get('nearest_any_D') or {}
    sites.append({
      'legacy_site_index':int(s['legacy_site_index']),
      'legacy_reduced_vertex':int(s['legacy_reduced_vertex']),
      'frozen_source_vertex_id':slot,
      'frozen_donor_body':int(s['donor_body']),
      'frozen_donor_name':expected_name,
      'frozen_side':expected_side,
      'stable_id':s['stable_id'],
      'original_kA_N_per_m':float(s['original_kA_N_per_m']),
      'frozen_target_world_m':s['target_world_in_069i_frame_m'],
      'current_emitted_slot_reused':True,
      'current_internal_new_vertex_id':new_internal,
      'current_ancestry_side':a_side,
      'current_ancestry_matches_frozen_side':bool(a_side==expected_side),
      'current_registered_dominant_donor_body':current_donor,
      'current_registered_dominant_donor_name':current_name,
      'current_registered_world_m':[float(row[k]) for k in ('wx','wy','wz')],
      'replacement_ancestry_chain':chain,
      'nearest_any_D_face':nearest.get('face'),
      'nearest_any_D_region_label':nearest.get('region'),
      'nearest_any_D_residual_m':nearest.get('residual_m'),
      'correspondence_status':'unresolved_no_record_emitted'
    })

reassigned=sum(x['current_emitted_slot_reused'] for x in sites)
cross_side=sum(not x['current_ancestry_matches_frozen_side'] for x in sites)
report={
 'schema':'numi.human.capitate.attachment.source-slot-provenance-audit.v1',
 'status':'rejected_correspondence_no_attachment_records_emitted',
 'purpose':'Trace frozen attachment source IDs through the replacement slot allocator; establish whether raw integer equality preserves anatomical source identity.',
 'inputs':{str(p):sha(p) for p in (BASE,CORR,REG,BINDINGS,G057,G069)},
 'source_map_equality':{'stage057_to_069i_internal_new_vertex_ids_equal':bool(np.array_equal(z57['internal_new_vertex_ids'],z69['internal_new_vertex_ids'])), 'stage057_to_069i_emitted_source_slots_equal':bool(np.array_equal(z57['emitted_source_slots'],z69['emitted_source_slots']))},
 'allocator_semantics':{
   'evidence_source':str(E/'finger-profile-060/reproduction-tools/rebuild_wrist_transition057.py'),
   'evidence_sha256':sha(E/'finger-profile-060/reproduction-tools/rebuild_wrist_transition057.py'),
   'mechanism':'The builder maps new internal vertex IDs into sorted baseline orphan source slots, then writes internal_new_vertex_ids and emitted_source_slots. A source slot number therefore ceases to identify the old vertex when present in emitted_source_slots.',
   'frozen_source_ids_found_in_emitted_source_slots':reassigned,
   'frozen_site_count':len(sites),
   'cross_side_reassignments':cross_side,
 },
 'frame_and_region_correction':{
   'donor_frames':'The preceding coordinate-frame audit found current CSV donor transforms for bodies 49/99 match the frozen matrices/translations; no frame mismatch established.',
   'donor_name_source':str(BINDINGS),
   'region_filter_correction':'The prior correspondence report wrist_carpals-only nearest search selected distant faces (about 0.48-0.51 m for some sites). This is a stale/incorrect region-filter result, not evidence of opposite-side anatomy. nearest_any_D fields are retained per site; they are not accepted attachment correspondences.',
 },
 'decision':'The six frozen records cannot be transferred by raw source_vertex equality: all six IDs are explicitly reused output slots. Five map to replacement ancestry on the side opposite the frozen capitate donor; the sole same-side case is still a generated replacement vertex with a different ancestry chain. Candidate source-identity preservation is therefore unproven. Do not create surface attachment records from nearest triangles or raw IDs; retain the six sites as unresolved until a unique original-identity or compiler-provenance mapping is recovered.',
 'sites':sites,
 'accepted_replacements':0,
 'notes':['All values are provenance/mapping diagnostics, not physical admission.','The original target world point, stable ID and kA are preserved exactly in this report.','This script is read-only with respect to application headers and emits no runtime candidate records.']
}
OUT.mkdir(parents=True,exist_ok=True)
output=OUT/'source-slot-provenance-audit.json'
output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
back=json.loads(output.read_text())
assert len(back['sites'])==6 and back['accepted_replacements']==0
assert back['allocator_semantics']['frozen_source_ids_found_in_emitted_source_slots']==6
print(json.dumps({'status':back['status'],'output':str(output),'output_sha256':sha(output),'slot_reassignments':reassigned,'cross_side_reassignments':cross_side,'site_mapping':[{k:v for k,v in x.items() if k in ('legacy_site_index','frozen_source_vertex_id','frozen_donor_name','current_internal_new_vertex_id','current_ancestry_side','current_registered_dominant_donor_name','replacement_ancestry_chain','nearest_any_D_residual_m')} for x in sites]},indent=2))
