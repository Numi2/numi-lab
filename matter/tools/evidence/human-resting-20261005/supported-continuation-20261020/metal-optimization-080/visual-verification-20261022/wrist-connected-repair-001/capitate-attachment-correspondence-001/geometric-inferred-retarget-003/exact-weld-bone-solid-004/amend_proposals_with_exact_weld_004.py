#!/usr/bin/env python3
"""Attach exact-weld solid classifications to existing points; no search or mutation of originals."""
import hashlib,json
from pathlib import Path
OUT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/capitate-attachment-correspondence-001/geometric-inferred-retarget-003')
BASE=OUT/'geometric-retarget-ladder.json'
WELD=OUT/'exact-weld-bone-solid-004/exact-weld-bone-witnesses.json'
MAP=OUT/'exact-weld-bone-solid-004/exact-weld-maps.npz'
OUTFILE=OUT/'geometric-retarget-ladder-exact-weld-amended.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
base=json.loads(BASE.read_text());weld=json.loads(WELD.read_text());by={int(s['legacy_site_index']):s for s in weld['sites']}
for site in base['sites']:
 diag=by[int(site['legacy_site_index'])];cls={n:v for n,v in diag['witnesses_by_same_side_bone'].items()}
 site['exact_weld_bone_reclassification']={'point_role':diag['point_role'],'deep_point_world_m':diag['deep_point_world_m'],'binding':diag['binding'],'witnesses_by_same_side_bone':cls,'source_report':str(WELD),'source_report_sha256':sha(WELD),'weld_map_npz':str(MAP),'weld_map_sha256':sha(MAP)}
 for rung in site['displacement_ladder']:
  p=rung.get('proposal')
  if p is None:continue
  aw=p['anatomical_witness']
  for entry in aw['bone_landmark_witnesses']:
   ex=cls.get(entry['name'])
   if ex is None:raise RuntimeError(f"missing welded witness for site {site['legacy_site_index']} bone {entry['name']}")
   entry['provisional_pre_weld_classification']=entry['solid_witness']
   entry['solid_witness']={**ex,'classification_source':'exact Float32 coordinate-welded closed bone surface'}
  for key in ('capitate_bone_witness','nearest_owner_bone_witness'):
   entry=aw.get(key)
   if entry is None:continue
   ex=cls.get(entry['name'])
   if ex is None:raise RuntimeError(f"missing exact-weld {key} for site {site['legacy_site_index']} {entry['name']}")
   entry['provisional_pre_weld_classification']=entry['solid_witness']
   entry['solid_witness']={**ex,'classification_source':'exact Float32 coordinate-welded closed bone surface'}
base['identity_result_nomenclature']={
 'source_identity_recovered':'0/6',
 'source_identity_evidence':'source-slot-provenance-002/source-slot-provenance-audit.json; accepted_replacements=0 and all six raw source IDs were reused output slots',
 'earlier_strict_inference':'1/6 proposal under the earlier inferred-attachment criterion; source-slot-provenance-002/inferred-capitate-retargeting-proposals.json',
 'current_geometric_screen':'5/6 candidate-only geometric proposals under the separate same-side D-surface screen; site16 remains outside its 20mm ladder',
 'distinction':'A new inferred geometric candidate is not recovered source identity.'}
base['exact_weld_bone_solid_followup']={'report':str(WELD),'report_sha256':sha(WELD),'weld_map_npz':str(MAP),'weld_map_sha256':sha(MAP),'interpretation':'All nine queried same-side carpal/metacarpal bones are closed and consistently oriented after exact registered-Float32 weld. The original report remains preserved; its provisional open-mesh witness fields are retained under provisional_pre_weld_classification and are superseded by solid_witness in this amended copy.'}
base['amended_from_report_sha256']=sha(BASE)
OUTFILE.write_text(json.dumps(base,indent=2,sort_keys=True,allow_nan=False)+'\n')
check=json.loads(OUTFILE.read_text());assert len(check['sites'])==6 and check['identity_result_nomenclature']['source_identity_recovered']=='0/6'
print(json.dumps({'output':str(OUTFILE),'sha256':sha(OUTFILE),'source_report_sha256':sha(BASE),'exact_weld_report_sha256':sha(WELD),'site_count':len(check['sites'])},indent=2))
