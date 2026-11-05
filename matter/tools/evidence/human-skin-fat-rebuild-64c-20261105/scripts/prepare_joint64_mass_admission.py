from pathlib import Path
import json,hashlib,copy,importlib.util,sys,csv,math,numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');C=W/'joint-boundary-rebuild-64c/compiler'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');B=H/'regional-compensation-1061-root'
out=C/'mass-admission';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
def put(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
rawrun=C/'mass-allocation/run-manifest.json';rawexec=C/'mass-allocation/execution.json'
r=json.loads(rawrun.read_text());env={a.split('=',1)[0]:a.split('=',1)[1] for a in r['argv'] if a.startswith('NUMI_H706_FRACTIONAL_')}
old=json.loads((B/'full-allocation-003/run-manifest.json').read_text());cfg=json.loads((B/'postpass-005/identity-003.json').read_text())
pre='NUMI_H706_FRACTIONAL_'
paths={ 'current_state':C/'volume-prefix/verified-state.json','candidate_skin_volume':C/'canonical/skin-canonical-parent-union.npz','candidate_fat_volume':C/'canonical/fat-canonical-parent-union.npz',
 'child_skin_node':C/'volume-export/volume/mesh.node','child_fat_node':C/'volume-export/volume/mesh.node','child_skin_elements':C/'volume-export/volume/mesh.ele','child_fat_elements':C/'volume-export/volume/mesh.ele','child_skin_labels':C/'volume-export/volume/child.labels.csv','child_fat_labels':C/'volume-export/volume/child.labels.csv',
 'projection_report':out/'projection-report.json'}
for k,v in {'p1_skin_support':'P1_SKIN_NPZ','p1_fat_support':'P1_FAT_NPZ','carrier_skin_packet':'CARRIERS_SKIN_NPZ','carrier_fat_packet':'CARRIERS_FAT_NPZ','projection_evidence':'EXACT_PROJECTION_CSV','source_centroid_slots':'SOURCE_CENTROID_CSV','carrier_witness_skin':'CARRIER_WITNESS_SKIN_CSV','carrier_witness_fat':'CARRIER_WITNESS_FAT_CSV'}.items():paths[k]=Path(env[pre+v])
assert sha(paths['current_state'])==env[pre+'STATE_SHA256']
put(paths['projection_report'],{'qualified_projection_csv':pin(paths['projection_evidence']),'producer':pin(W/'rebind_joint64_mass_support.py'),'support_preparation_receipt':pin(C/'mass-support/receipt.json'),'physical_admission':False,'native_steps':0})
named={k:pin(paths[k]) if k in paths else v for k,v in old['named_input_pins'].items()}
# Publish a replay context for the completed raw invocation; the raw evidence is immutable.
run={k:copy.deepcopy(old[k]) for k in ['schema','variant','source','binary','source_closure','source_control_manifest','file_pins']};run.update(argv=r['argv'],output_dir=env[pre+'OUTPUT_DIR'],named_input_pins=named,run_config=pin(C/'mass-allocation/run-config.json'),stdout_log=str(C/'mass-allocation/stdout.log'),stderr_log=str(C/'mass-allocation/stderr.log'),physical_admission=False,native_steps=0,raw_invocation=pin(rawrun),raw_execution=pin(rawexec))
# No runtime or source-policy changes are made by this postpass.
put(out/'run-manifest.json',run)
execution=json.loads(rawexec.read_text());assert execution['returncode']==0
put(out/'execution.json',{'exit_code':execution['returncode'],'manifest_sha256':sha(out/'run-manifest.json'),'raw_execution':pin(rawexec),'physical_admission':False,'native_steps':0})
remap={old['named_input_pins'][k]['path']:named[k]['path'] for k in named}
oldE=H/'current-native-1024-root/current-433-1044/volume-export';newE=C/'volume-export'
# Rebind the raw export inventory and source-only proof paths.
for p in newE.rglob('*'):
 if p.is_file():remap[str(oldE/p.relative_to(newE))]=str(p)
oldgeom=cfg['geometry_admission'];cfg['geometry_admission']=json.loads((C/'source-geometry-admission-spec-001.json').read_text())
for key,v in cfg['native_preflight_env'].items():
 if v in remap:cfg['native_preflight_env'][key]=remap[v]
def refresh_hashes(env):
 for k,v in list(env.items()):
  if not k.endswith('SHA256'):continue
  options=[k[:-7],k[:-7]+'_CSV',k[:-7]+'_ASSET']
  for pk in options:
   if pk in env and Path(env[pk]).is_file():env[k]=sha(env[pk]);break
refresh_hashes(cfg['native_preflight_env'])
# Candidate rich volume and typed JSON inventories live under different historical roots.
ne=cfg['native_preflight_env']
for k,p in [('OPERATIONAL_CANDIDATE_VOLUME_ASSET',C/'rich433/candidate-current-typed-volume.npz'),('OPERATIONAL_CANDIDATE_PRODUCER_RECEIPT',newE/'receipt.json'),('OPERATIONAL_CANDIDATE_SELF_AUDIT_RECEIPT',C/'source-geometry-admission-001.json')]:
 ne[k]=str(p);hk=k+'_SHA256' if not k.endswith('_ASSET') else 'OPERATIONAL_CANDIDATE_VOLUME_SHA256';ne[hk]=sha(p)
export=json.loads((newE/'receipt.json').read_text())
for role in 'OID':
 ne['ROLE_'+role+'_SOURCE_FACE_COUNT']=str(export['role_faces'][role]);ne['ROLE_'+role+'_SOURCE_VERTEX_COUNT']=str(export['role_nodes'][role])
ne['OPERATIONAL_CONFORMING_REFINEMENT_CHAIN']='0'
cfg['h706_fractional_env']={k:env[pre+k] for k in cfg['h706_fractional_env']}
for k in cfg['h706_fractional_path_pins']:
 p=cfg['h706_fractional_path_pins'][k]['path']
 if p in remap:p=remap[p]
 cfg['h706_fractional_path_pins'][k]={'path':p,'resolved_path':str(Path(p).resolve()),'sha256':sha(p)}
cfg['h706_fractional_environment_source']={'path':str(out/'run-manifest.json'),'resolved_path':str(out/'run-manifest.json'),'sha256':sha(out/'run-manifest.json'),'argv_sha256':hashlib.sha256(json.dumps(run['argv'],sort_keys=True,separators=(',',':')).encode()).hexdigest()}
cfg['h706_fractional_run_control_pins']={k:{t:run[k][t] for t in ['path','sha256']} for k in ['run_config','source_control_manifest']}
cfg['h706_fractional_auxiliary_path_pins']['projection_report']=dict(pin(paths['projection_report']),resolved_path=str(paths['projection_report']))
cfg['input_pins'].update(json.loads((C/'source-geometry-candidate-input-pins-001.json').read_text()))
for v in named.values():cfg['input_pins'][v['path']]=v['sha256']
for k,v in cfg['h706_fractional_path_pins'].items():cfg['input_pins'][v['path']]=v['sha256']
# Derive the same geometry envelope from the new P1 centroid assignment.
src=Path(named['source330_mass']['path']);rows=list(csv.DictReader(Path(named['source330_nodes']['path']).open()))
pv=np.array([[float(r[k]) for k in ['current_x_m','current_y_m','current_z_m']] for r in rows])
rows=list(csv.DictReader(Path(named['source330_tets']['path']).open()));pt=np.array([[int(r['parent_node'+str(i)]) for i in range(4)] for r in rows]);pm=np.array([int(r['material_index']) for r in rows]);pc=pv[pt].mean(1)
with np.load(C/'rich433/candidate-current-typed-volume.npz',allow_pickle=False) as z:
 x=z['points_reference_world_m_f64'];t=z['volume_tetrahedra'];p=z['child_parent_tet_row'];m=z['material_index'];assert np.array_equal(p,z['current_typed_p1_anchor_selected_parent_tet_row']) and np.array_equal(pm[p],m)
 d=np.linalg.norm(x[t].mean(1)-pc[p],axis=1)
mx=float(d.max());cap=max(200,math.ceil(mx*1e6)+1)/1e6
assert cap==.187444,('new geometry bound differs from the unchanged allocator bound',mx,cap)
frame=src.parent/'registered-skin-poses-100.json'
pins={'state':named['current_state'],'candidate':pin(C/'rich433/candidate-current-typed-volume.npz'),'child_node':named['child_skin_node'],'child_element':named['child_skin_elements'],'child_labels':named['child_skin_labels'],'coordinate_frame':pin(frame),'source330_mass':pin(src),'source330_source':pin(Path(str(src)+'.source.csv')),'source330_nodes':named['source330_nodes'],'source330_tets':named['source330_tets'],'source330_labels':named['source330_labels'],'geometry_skin':named['candidate_skin_volume'],'geometry_fat':named['candidate_fat_volume']}
receipt=copy.deepcopy(export);receipt['mass_transport_envelope']={'schema':'numi.human.mass-transport-envelope.v1','status':'PREDECLARED_CURRENT_GEOMETRY_ENVELOPE','rule_id':'ceil_max_declared_source_centroid_um_plus_1_floor_200','maximum_declared_source_centroid_distance_m':mx,'predeclared_transport_envelope_m':cap,'exact_decimal':format(cap,'.6f'),'input_sha256':{k:v['sha256'] for k,v in pins.items()}}
receipt['mass_mapping_maximum_transport_distance_m']=cap;receipt['input_sha256'].update({v['path']:v['sha256'] for v in pins.values()});put(out/'geometry-envelope.json',receipt)
cfg['native_preflight_env']['MASS_TRANSPORT_ENVELOPE_RECEIPT']=str(out/'geometry-envelope.json');cfg['native_preflight_env']['MASS_TRANSPORT_ENVELOPE_RECEIPT_SHA256']=sha(out/'geometry-envelope.json');cfg['mass_transport_envelope_receipt']=pin(out/'geometry-envelope.json');cfg['input_pins'][str(out/'geometry-envelope.json')]=sha(out/'geometry-envelope.json')
for v in pins.values():cfg['input_pins'][v['path']]=v['sha256']
cfg['h706_fractional_source_preflight']={'status':'PASS_CURRENT_INPUTS_PINNED','physical_admission':False,'native_steps':0}
cfg['candidate_label']='joint64c connected surface/volume contraction';cfg['output_root']=str(out/'unused-native-output')
put(out/'identity.json',cfg)
print('mass_identity_prepared',len(named),'maximum_source_centroid_distance_m',mx,flush=True)
# Keep the existing immutable numerical replay; extend only explicit receiver-row binding.
owner=B/'postpass-005/build_current_fractional_mass_receipt_1061.py';s=owner.read_text()
a=s.index("        if len(packet['regional_compensation_child_rows'])!=6359")
b=s.index("    out=dict(control)",a)
s=s[:a]+"""        binding_path=Path("""+repr(str(C/'mass-support/receiver-rebind.npz'))+""")
        if sha_file(binding_path)!="""+repr(sha(C/'mass-support/receiver-rebind.npz'))+""":raise core.ReceiptError('receiver-rebind pin changed')
        with np.load(binding_path,allow_pickle=False) as maps, np.load("""+repr(str(B/'skin-source-carriers-with-regional-compensation-002.npz'))+""",allow_pickle=False) as baseline:
            # targets() uses the previously authenticated old-row receiver mapping.
            offsets=maps['old_to_new_offsets'];children=maps['old_to_new_rows']
            expected=np.unique(np.concatenate([children[offsets[int(i)]:offsets[int(i)+1]] for i in baseline['regional_compensation_child_rows']]))
            if not np.array_equal(packet['regional_compensation_child_rows'],expected):raise core.ReceiptError('compensation receiver set is not exactly remapped old scope')
            if packet['regional_compensation_limits'].tolist()!=baseline['regional_compensation_limits'].tolist():raise core.ReceiptError('mass or distance exception changed')
""" +s[b:]
adapter=out/'build_current_fractional_mass_receipt_1061.py';adapter.write_text(s)
# Record explicit context/provenance before replay.
put(out/'preparation.json',{'raw_invocation':pin(rawrun),'raw_execution':pin(rawexec),'identity':pin(out/'identity.json'),'existing_builder':pin(owner),'current_builder':pin(adapter),'geometry_envelope_maximum_m':mx,'physical_admission':False,'native_steps':0})
