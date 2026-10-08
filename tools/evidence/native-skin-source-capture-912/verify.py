from pathlib import Path
import json,hashlib,sys
E=Path('/Users/n/numi-human-resting-evidence-20261005')
O=E/'native-skin-source-capture-review-912';O.mkdir(exist_ok=True)
run=E/'native-skin-source-capture-912';ref=E/'native-composed-resting-cycle-909'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((run/'run-metadata.json').read_text());assert meta['exit_code']==0 and not meta['source_files_changed_during_run'] and meta['loaded_metal_runtime']['verified']
traces={}
for name in ['resting-coupled.csv','resting-com-q-integration.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv']:
 a=(run/name).read_bytes().splitlines(keepends=True);b=(ref/name).read_bytes().splitlines(keepends=True)
 same=a==b[:len(a)]; assert same,name
 traces[name]={'rows':len(a)-1,'exact_prefix_of_909':same,'sha256':sha(run/name),'reference_sha256':sha(ref/name)}
sys.path.insert(0,'/Users/n/numi-human-lung-source-publication-902/src')
from numilab_human.skin_source_payload_preflight import decode_payload
skin=E/'common-atlas-skin-composition-907/bodyparts3d-myosim-skinned-shell.nhskin'
d=decode_payload(skin.read_bytes());owners=set(map(int,d['bindings_u'][:,0]))
rows=[]
for step in [0,63]:
 p=run/f'accepted-geometry/step-{step}.receipt.json';r=json.loads(p.read_text())
 ids=[int(v['body_index']) for v in r['accepted_registered_body_poses']]
 assert len(ids)==len(set(ids)) and owners.issubset(ids)
 m=r['skin_source_mapping'];assert m['source_vertex_count']==int(d['vertex_count'])
 assert m['vertex_map_record_stride_bytes']==48
 for key in ['vertex_map','anatomy_parameters']:
  a=m[key];f=Path(a['path']);assert f.stat().st_size==a['bytes'] and sha(f)==a['sha256']
 row={'step':step,'receipt_sha256':sha(p),'skin_binding_owner_count':len(owners),'accepted_body_pose_count':len(ids),'all_skin_owners_covered_once':True,'skin_source_mapping':m,'respiratory_motion':r['accepted_respiratory_motion']}
 if step==0:
  b=json.loads((ref/'accepted-geometry/step-0.receipt.json').read_text())
  keys=['captured_vertex_buffer_sha256','accepted_body_state_sha256','accepted_respiration_state_sha256','accepted_root_fingerprint','accepted_transaction_fingerprint']
  row['exact_step0_reference_keys']={k:r[k]==b[k] for k in keys};assert all(row['exact_step0_reference_keys'].values())
 rows.append(row)
report={'scope':'Native source-state capture regression; geometry and physical states unchanged over64steps. Not a full breathing cycle or endurance qualification.','native_exit_code':0,'wall_seconds':meta['wall_seconds'],'traces':traces,'captures':rows,'native_sha256':sha(Path('/Users/n/numi-human-source-capture-build-016/bin/numi-human-native')),'build_manifest_sha256':sha(Path('/Users/n/numi-human-source-capture-build-016/evidence/build-manifest.json'))}
(O/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'traces':{k:v['rows'] for k,v in traces.items()},'skin_owners':len(owners),'captured_owners':len(ids),'step0_same':rows[0]['exact_step0_reference_keys'],'wall_s':meta['wall_seconds']},indent=2))
