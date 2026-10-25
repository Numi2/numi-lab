#!/usr/bin/env python3
"""Read-only exact-witness ownership map for frozen 030 edge reports."""
import collections, hashlib, json
from pathlib import Path
import numpy as np

E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
R=E/'route-audit-counter-fix-030'
OUT=R/'regional-map-031'
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(f'refusing to overwrite {OUT}')
OUT.mkdir(parents=True, exist_ok=True)
REPORTS={
 'baseline035':R/'baseline035/tet-edge-report.json',
 'candidate014':R/'candidate014/tet-edge-report.json',
 'candidate027':R/'candidate027/tet-edge-report.json',
}
EXPECTED={
 REPORTS['baseline035']:'613d19fe45e052697ae5cdf8e7b481bfb3ea3fdbc5a4517d8e0e73aa53c40a23',
 REPORTS['candidate014']:'0ad07a3393aa4ba715f17e66fc70bd75f712cf2c39d1f027301ff2788335effe',
 REPORTS['candidate027']:'6325ea1b5c23f31d4b25cc1a22d768e69ff475d08a1cf12d98d7b62ae83e9fa0',
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for p,want in EXPECTED.items():
 got=sha(p)
 if got!=want:raise RuntimeError(f'frozen report hash mismatch: {p}: {got} != {want}')
reports={name:json.loads(p.read_text()) for name,p in REPORTS.items()}
base_npz=R/'baseline-035.npz'
base_sha=sha(base_npz)
z=np.load(base_npz,allow_pickle=False)
x0=z['points_world_f32']; gids=z['global_node_ids']; T=z['tetrahedra'].astype(np.int64); mats=z['material_ids']
N=2411
candidates={
 'baseline035':x0,
 'candidate014':np.load(E/'baseline-depth-transition-014/candidate.npz',allow_pickle=False)['points_world_f32'],
 'candidate027':np.load(E/'o12-outward-repair-027-f32-gated/candidate.npz',allow_pickle=False)['points_world_f32'],
}
assert x0.shape==(7233,3) and gids.shape==(7233,) and T.shape==(28632,4)
for name,x in candidates.items():
 if x.shape!=x0.shape or x.dtype!=np.float32:raise RuntimeError(f'{name} node array shape/dtype mismatch')
if not np.array_equal(T,np.load(E/'baseline-depth-transition-014/candidate.npz',allow_pickle=False)['tetrahedra']):raise RuntimeError('014 topology differs')
if not np.array_equal(T,np.load(E/'o12-outward-repair-027-f32-gated/candidate.npz',allow_pickle=False)['tetrahedra']):raise RuntimeError('027 topology differs')
if not np.array_equal(mats,np.load(E/'baseline-depth-transition-014/candidate.npz',allow_pickle=False)['material_ids']):raise RuntimeError('014 materials differ')
if not np.array_equal(mats,np.load(E/'o12-outward-repair-027-f32-gated/candidate.npz',allow_pickle=False)['material_ids']):raise RuntimeError('027 materials differ')
# Real incident material IDs for each FEM edge.
edge_materials=collections.defaultdict(set)
for tet,material in zip(T,mats):
 for i in range(4):
  for j in range(i+1,4):edge_materials[tuple(sorted((int(tet[i]),int(tet[j]))))].add(int(material))
def layerpair_name(lp):
 return '-'.join('OID'[i] for i in lp)
def group_map(name):
 out={}
 for w in reports[name]['witnesses']:
  local=tuple(map(int,w['endpoint_local_nodes']))
  global_nodes=tuple(map(int,w['endpoint_global_nodes']))
  if any(i<0 or i>=len(gids) for i in local):raise RuntimeError(f'bad local endpoint row in {name}')
  if tuple(map(int,gids[list(local)]))!=global_nodes:raise RuntimeError(f'global/local endpoint mismatch in {name}: {local}/{global_nodes}')
  layers=tuple(sorted(map(int,w['endpoint_layers'])))
  if layers!=tuple(sorted(i//N for i in local)):raise RuntimeError(f'layer/local endpoint mismatch in {name}')
  columns=tuple(map(int,w['endpoint_columns']))
  if columns!=tuple(i%N for i in local):raise RuntimeError(f'column/local endpoint mismatch in {name}')
  key=(tuple(sorted(global_nodes)),int(w['body']),int(w['bone_stable_id']))
  edge=tuple(sorted(local)); material_ids=tuple(sorted(edge_materials.get(edge,())))
  if not material_ids:raise RuntimeError(f'edge absent from tet material domain: {edge}')
  changed=bool(np.any(candidates[name][local[0]]!=x0[local[0]]) or np.any(candidates[name][local[1]]!=x0[local[1]]))
  meta={'edge_global_nodes':key[0],'edge_local_nodes':edge,'body':key[1],'bone_stable_id':key[2],
        'layers':layers,'layer_pair':layerpair_name(layers),'incident_material_ids':material_ids,
        'source_columns':tuple(sorted(columns)),'source_columns_by_layer':tuple(sorted((int(i%N),int(i//N)) for i in local)),'changed_endpoint_vs_baseline':changed,
        'triangle_stable_ids':set()}
  if key in out:
   old=out[key]
   for field in ('edge_local_nodes','layers','incident_material_ids','source_columns','changed_endpoint_vs_baseline'):
    if old[field]!=meta[field]:raise RuntimeError(f'inconsistent group metadata {name} {key} field {field}')
  else:out[key]=meta
  out[key]['triangle_stable_ids'].add(int(w['triangle_stable_id']))
 for v in out.values():v['triangle_stable_ids']=tuple(sorted(v['triangle_stable_ids']))
 return out
G={n:group_map(n) for n in reports}
def cat_counts(d):
 c=collections.Counter((v['layer_pair'],v['incident_material_ids']) for v in d.values())
 return [{'layer_pair':k[0],'material_ids':list(k[1]),'groups':n} for k,n in sorted(c.items())]
def bank_counts(d):
 c=collections.Counter((v['body'],v['bone_stable_id']) for v in d.values())
 return [{'body_index':k[0],'bone_stable_id':k[1],'groups':n} for k,n in c.most_common()]
def bank_source_columns(d):
 by=collections.defaultdict(lambda:{'groups':0,'layer_material':collections.Counter(),'source_columns_by_layer':collections.defaultdict(set)})
 for v in d.values():
  item=by[(v['body'],v['bone_stable_id'])];item['groups']+=1;item['layer_material'][(v['layer_pair'],v['incident_material_ids'])]+=1
  for col,layer in v['source_columns_by_layer']:item['source_columns_by_layer']['OID'[layer]].add(col)
 return [{'body_index':k[0],'bone_stable_id':k[1],'groups':v['groups'],'layer_material_groups':[{'layer_pair':pair[0],'material_ids':list(pair[1]),'groups':n} for pair,n in sorted(v['layer_material'].items())], 'source_columns_by_layer':{layer:sorted(cols) for layer,cols in sorted(v['source_columns_by_layer'].items())}} for k,v in sorted(by.items())]
def delta(a,b):
 ga,gb=G[a],G[b]; add=set(gb)-set(ga); remove=set(ga)-set(gb); keep=set(ga)&set(gb)
 ag={k:gb[k] for k in add}; rg={k:ga[k] for k in remove}
 cols_by_layer=collections.defaultdict(set)
 for v in ag.values():
  for col,layer in v['source_columns_by_layer']:cols_by_layer['OID'[layer]].add(col)
 return {'added_groups':len(add),'removed_groups':len(remove),'retained_groups':len(keep),
         'added_groups_by_layer_pair_material':cat_counts(ag),'removed_groups_by_layer_pair_material':cat_counts(rg),
         'added_groups_top_banks':bank_counts(ag)[:20],'removed_groups_top_banks':bank_counts(rg)[:20],
         'added_source_columns_by_layer':{k:sorted(v) for k,v in sorted(cols_by_layer.items())},
         'removed_source_columns_by_layer':{k:sorted({col for v in rg.values() for col,layer in v['source_columns_by_layer'] if 'OID'[layer]==k}) for k in 'OID'},
         'added_bank_source_columns':bank_source_columns(ag),'removed_bank_source_columns':bank_source_columns(rg)}
patch_report=json.loads((E/'o12-outward-repair-027-f32-gated/report.json').read_text())
patch_columns=sorted(map(int,patch_report['construction']['shape_columns']))
changed_rows={n:np.any(x!=x0,axis=1) for n,x in candidates.items()}
changed_columns={n:sorted(set((np.flatnonzero(changed_rows[n])%N).tolist())) for n in candidates}
changed_027_014=sorted(set((np.flatnonzero(np.any(candidates['candidate027']!=candidates['candidate014'],axis=1))%N).tolist()))
current=G['candidate027']
changed_group_count=sum(v['changed_endpoint_vs_baseline'] for v in current.values())
summary={
 'schema':'numi.human.tet-edge-witness-regional-map.v1',
 'status':'READ_ONLY_DIAGNOSTIC_NO_CLEARANCE_OR_PHYSICAL_ADMISSION',
 'sources':{
  'baseline_candidate_npz':{'path':str(base_npz),'sha256':base_sha},
  'baseline_report':{'path':str(REPORTS['baseline035']),'sha256':EXPECTED[REPORTS['baseline035']]},
  'candidate014_report':{'path':str(REPORTS['candidate014']),'sha256':EXPECTED[REPORTS['candidate014']]},
  'candidate027_report':{'path':str(REPORTS['candidate027']),'sha256':EXPECTED[REPORTS['candidate027']]},
  'candidate014_npz':{'path':str(E/'baseline-depth-transition-014/candidate.npz'),'sha256':sha(E/'baseline-depth-transition-014/candidate.npz')},
  'candidate027_npz':{'path':str(E/'o12-outward-repair-027-f32-gated/candidate.npz'),'sha256':sha(E/'o12-outward-repair-027-f32-gated/candidate.npz')},
  'comparison_json':{'path':str(R/'comparison.json'),'sha256':sha(R/'comparison.json')},
  'topology':'Tetrahedra/material arrays are byte-value equal across all three pinned candidate payloads. Baseline global IDs are checked against every witness local/global endpoint.'},
 'domain_counts':{n:{'proper_triangle_hits':len(reports[n]['witnesses']),
                      'unique_edges_with_hits':reports[n]['counts']['unique_tetrahedral_edges'] if False else reports[n]['edges_with_hits'],
                      'edge_bank_groups':reports[n]['edge_bank_groups_with_hits'],
                      'groups_by_layer_pair_material':cat_counts(G[n]),'typed_banks':bank_source_columns(G[n])} for n in G},
 'group_set_differences':{
   'baseline035_to_candidate014':delta('baseline035','candidate014'),
   'candidate014_to_candidate027':delta('candidate014','candidate027'),
   'baseline035_to_candidate027':delta('baseline035','candidate027')},
 'candidate027_patch_scope':{
   'optimizer_shape_columns':patch_columns,
   'changed_source_columns_vs_baseline_count':len(changed_columns['candidate027']),
   'changed_source_columns_vs_baseline':changed_columns['candidate027'],
   'changed_source_columns_vs_014_count':len(changed_027_014),
   'changed_source_columns_vs_014':changed_027_014,
   'groups_with_any_endpoint_changed_vs_baseline':changed_group_count,
   'groups_with_both_endpoints_unchanged_vs_baseline':len(current)-changed_group_count,
   'changed_groups_by_layer_pair_material':cat_counts({k:v for k,v in current.items() if v['changed_endpoint_vs_baseline']}),
   'unchanged_groups_by_layer_pair_material':cat_counts({k:v for k,v in current.items() if not v['changed_endpoint_vs_baseline']})},
 'interpretation':{
  'material_ids':{'1':'skin O-I material','2':'fat I-D material'},
  'layer_indices':{'0':'O outer','1':'I skin-fat interface','2':'D inner'},
  'limitation':'An edge hit is a finite FEM tetrahedral edge intersecting a registered typed-bone triangle. It is not a tet/bone volume-overlap count or a full surface-clearance certificate. Added/removed group counts compare exact typed bank and global FEM edge identities.'}}
OUT.mkdir(parents=True,exist_ok=True)
report_path=OUT/'regional-map.json'
report_path.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'out':str(report_path),'sha256':sha(report_path),'groups':{n:len(G[n]) for n in G},'delta_baseline_to_027':summary['group_set_differences']['baseline035_to_candidate027'],'candidate027_patch_scope':summary['candidate027_patch_scope']},indent=2))
