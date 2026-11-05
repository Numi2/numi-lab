from pathlib import Path
import json,hashlib,shutil,subprocess
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');C=W/'joint-boundary-rebuild-64c/compiler'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');B=H/'regional-compensation-1061-root'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def put(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
source=W/'shared-bundle-native-307';out=W/'joint-boundary-native-308';out.mkdir(exist_ok=False)
plan=json.loads((source/'install-motion.plan.json').read_text());cfg=json.loads((C/'mass-admission/identity.json').read_text())
conv=json.loads((C/'converter/execution.json').read_text());assert conv['exit_code']==0
stage=plan['stages'][0];oldroot=stage['argv'][4];stage['argv'][4]=str(out/'install-motion')
env=stage['environment'];mapping={}
oldrun=json.loads((B/'full-allocation-003/run-manifest.json').read_text());newrun=json.loads((C/'mass-admission/run-manifest.json').read_text())
for k,v in oldrun['named_input_pins'].items():mapping[v['path']]=newrun['named_input_pins'][k]['path']
for olddir,newdir in [(H/'current-native-1024-root/current-433-1044/volume-export',C/'volume-export'),(B/'converter-020/role-output',C/'converter/role-output'),(B/'full-allocation-003/allocator-output',C/'mass-allocation/allocator-output')]:
 for p in newdir.rglob('*'):
  if p.is_file():mapping[str(olddir/p.relative_to(newdir))]=str(p)
mapping[str(B/'postpass-005/mass-receipt-007.json')]=str(C/'mass-admission/mass-receipt.json')
mapping[str(H/'current-native-1024-root/current-geometry-envelope-1058/run-002/receipt.json')]=str(C/'mass-admission/geometry-envelope.json')
for p in (C/'bindings').iterdir():
 if p.name in ['registered-O.csv','registered-I.csv','registered-D.csv','O-source.json','I-source.json','D-source.json','combined-node-map.csv']:
  mapping[str(B/'dominant-frame-registration-042'/p.name)]=str(p)
for key,p in [('registration_restoration',C/'bindings/registration-restoration.json'),('attachment_remap',C/'bindings/remap-receipt.json'),('current_frame_registration',C/'bindings/receipt.json')]:
 old=plan[key]['path'];mapping[old]=str(p);plan[key]=ref(p)
mapping[str(B/'attachment-remap-039/deep-attachment-requests.csv')]=str(C/'bindings/deep-attachment-requests.csv')
mapping[str(B/'current-reference-import-031/complete-node-parent-registration.csv')]=str(C/'bindings/complete-node-parent-registration.csv')
pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
for k,v in list(env.items()):
 if v in mapping:env[k]=mapping[v]
 if v.startswith(oldroot):env[k]=str(out/'install-motion')+v[len(oldroot):]
for k,v in cfg['h706_fractional_env'].items():env['NUMI_H706_FRACTIONAL_'+k]=v
# Use current complete bindings after source-only mass conversion.
for k,p in [('OPERATIONAL_CANDIDATE_VOLUME_ASSET',C/'rich433/candidate-current-typed-volume.npz'),('OPERATIONAL_CANDIDATE_PRODUCER_RECEIPT',C/'volume-export/receipt.json'),('OPERATIONAL_CANDIDATE_SELF_AUDIT_RECEIPT',C/'source-geometry-admission-001.json'),('COMBINED_NODE_MAP_CSV',C/'bindings/combined-node-map.csv'),('NODE_REGISTRATION_CSV',C/'bindings/complete-node-parent-registration.csv'),('DEEP_ATTACHMENT_REQUESTS_CSV',C/'bindings/deep-attachment-requests.csv')]:
 env[pre+k]=str(p)
export=json.loads((C/'volume-export/receipt.json').read_text())
for role in 'OID':
 env[pre+'ROLE_'+role+'_SOURCE_ASSET']=str(C/f'bindings/{role}-source.json')
 env[pre+'SURFACE_'+role+'_NODES_CSV']=str(C/f'bindings/registered-{role}.csv')
 env[pre+'ROLE_'+role+'_SOURCE_FACE_COUNT']=str(export['role_faces'][role])
 env[pre+'ROLE_'+role+'_SOURCE_VERTEX_COUNT']=str(next(item['source_point_domain_count'] for item in json.loads((C/'volume-export/source-domain-registry.json').read_text())['roles'] if item['role']==role))
env[pre+'OPERATIONAL_CONFORMING_REFINEMENT_CHAIN']='0'
# The new compiled mesh already includes the accepted conditioning cavities.
removed={k:env.pop(k) for k in list(env) if k in ['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ','NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']}
for k,v in list(env.items()):
 if not k.endswith('SHA256'):continue
 for pk in [k[:-7],k[:-7]+'_CSV',k[:-7]+'_ASSET',k[:-7]+'_NPZ',k[:-7]+'_JSON']:
  if pk in env and Path(env[pk]).is_file():env[k]=sha(env[pk]);break
plan['geometry_admission']=json.loads((C/'source-geometry-admission-spec-001.json').read_text());plan['mass_admission']=ref(C/'mass-admission/mass-receipt.json')
plan['converter_run']=ref(C/'converter/run-manifest.json');plan['converter_execution']=ref(C/'converter/execution.json')
plan['source_qualifier']=ref(C/'source-geometry-admission-001.json')
# Preserve old source lineage separately, while updating actual current-input inventories.
plan['retained_prior_plan']=ref(source/'install-motion.plan.json')
def remap_inventory(value):
 if isinstance(value,list):return [remap_inventory(v) for v in value]
 if isinstance(value,dict):
  if isinstance(value.get('path'),str) and value['path'] in mapping:
   v=dict(value);v.update(ref(mapping[value['path']]));return v
  result={}
  for k,v in value.items():
   nk=mapping.get(k,k)
   if nk!=k and isinstance(v,str) and len(v)==64:result[nk]=sha(nk)
   elif nk!=k and isinstance(v,dict) and 'sha256' in v:result[nk]=dict(v,**ref(nk))
   else:result[nk]=remap_inventory(v)
  return result
 return value
for k in ['input_pins','role_input_pins','required_transitive_mass_proofs','required_transitive_input_pins']:
 plan[k]=remap_inventory(plan[k])
plan['source_preflight_reuse']={'reused':False,'reason':'new compiled connected surface/volume mesh and conservative mass allocation require full admission'}
plan['current_source_domain_audit']={'status':'PASS_EXACT_CURRENT_TYPED_COORDINATE_AND_SIGNED_FACE_BINDING_SOURCE_ONLY','receipt':ref(C/'volume-export/receipt.json'),'orientation':ref(C/'role-orientation.json'),'controls':ref(C/'signed-face-controls-receipt.json')}
plan['purpose']='Matched complete-scene test of joint64c connected skin/fat mechanics; surface-only and preparation results are not performance claims'
plan['trial_scope']='32 roots at unchanged 15.625us; no supported-rest, sustained physiology or presentation claim'
plan['predeclared_checks']={'reference':ref(source/'install-motion.plan.json'),'accepted_roots':32,'unchanged_dt_s':.000015625,'prediction':'Removing a limiting fold and low-quality cells may reduce late-root nonlinear work; no forecast whole-run speedup','primary_outcome':'accepted simulated time / measured body wall time','secondary':['active columns and budget exhaustion','process/GPU working set','contact/element certificates'],'fidelity':{'integrated_force_relative_tolerance':.01,'physiological_relative_tolerance':.01,'regional_energy_strain_relative_tolerance':.05,'unchanged_contact_penetration_limits':True,'mass_conservation':True,'comparison_mapping':'canonical cell ancestry; never raw changed indices'},'failure_rule':'retain failed or slower outcome; no relaxed certificate, density or physical gates'}
plan['already_compiled_conditioning']={'removed_reapplication':removed,'evidence':ref(W/'joint-boundary-rebuild-64c/report.json'),'reason':'candidate input composes accepted134 connectivity with connected64c quotient'}
for p in [C/'mass-admission/mass-receipt.json',C/'source-geometry-admission-001.json',C/'bindings/receipt.json',C/'bindings/remap-receipt.json',C/'bindings/registration-restoration.json']:
 if isinstance(plan['input_pins'],dict):plan['input_pins'][str(p)]=ref(p) if isinstance(next(iter(plan['input_pins'].values())),dict) else sha(p)
put(out/'install-motion.plan.json',plan);shutil.copy2(source/'run_native_plan.py',out/'run_native_plan.py')
print('plan_prepared',out,'free_gib',shutil.disk_usage(out).free/2**30)
# Pin every file-valued native input and paired SHA before launch.
issues=[]
for k,v in env.items():
 if k.endswith('SHA256'):
  for pk in [k[:-7],k[:-7]+'_CSV',k[:-7]+'_ASSET',k[:-7]+'_NPZ',k[:-7]+'_JSON']:
   if pk in env and Path(env[pk]).is_file():
    if sha(env[pk])!=v:issues.append((k,pk))
    break
assert not issues,issues
put(out/'prelaunch-input-checks.json',{'paired_hash_mismatches':issues,'physical_equations_changed':False,'native_steps':0,'physical_admission':False,'plan':ref(out/'install-motion.plan.json')})
