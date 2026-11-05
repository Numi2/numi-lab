import sys
from pathlib import Path
import json,hashlib,importlib.util,sys,numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else W/'joint-boundary-rebuild-64c';C=D/'compiler'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class Arrays(dict):
 @property
 def files(self):return list(self)
 def __enter__(self):return self
 def __exit__(self,*a):pass
typed=load(C/'typed-adapter/typed-current-schema.py','joint64typed')
orientation=load(C/'typed-adapter/audit-role-orientation.py','joint64orient')
tests=[]
def reject(name,fn):
 try:fn()
 except (ValueError,IndexError,KeyError,AssertionError) as e:tests.append({'name':name,'status':'PASS','rejection':str(e)});return
 raise AssertionError(name+' was accepted')
with np.load(D/'candidate-I-typed-current.npz',allow_pickle=False) as z:arr=Arrays({k:z[k] for k in z.files})
typed._int_array(arr,'quality_star_face_output_to_input_row');tests.append({'name':'valid_quotient_replay','status':'PASS'})
bad=Arrays(arr);bad['typed_parent_candidate_sha256']=np.asarray('0'*64);reject('stale_parent_hash',lambda:typed._int_array(bad,'quality_star_face_output_to_input_row'))
bad=Arrays(arr);bad['source_input_face_rows']=arr['source_input_face_rows'].copy();bad['source_input_face_rows'][1]=bad['source_input_face_rows'][0];reject('duplicate_parent_face_row',lambda:typed._int_array(bad,'quality_star_face_output_to_input_row'))
bad=Arrays(arr);bad['source_input_vertex_quotient']=arr['source_input_vertex_quotient'].copy();bad['source_input_vertex_quotient'][int(arr['faces'][0,0])]=len(arr['points_world_m_f32'])+1;reject('out_of_domain_quotient',lambda:typed._int_array(bad,'quality_star_face_output_to_input_row'))
bad=Arrays(arr);bad['faces']=arr['faces'].copy();bad['faces'][0]=bad['faces'][0,[0,2,1]];reject('nonreplaying_winding',lambda:typed._int_array(bad,'quality_star_face_output_to_input_row'))
with np.load(C/'shells/skin-shell.npz',allow_pickle=False) as z:skin=Arrays({k:z[k] for k in z.files})
with np.load(C/'shells/fat-shell.npz',allow_pickle=False) as z:fat=Arrays({k:z[k] for k in z.files})
state=json.loads((C/'verified-state.json').read_text())
for role in 'OID':
 r=orientation.check_asset(role,state['source_surfaces'][role],skin,fat)
 assert r['source_face_ancestry_exact'] and all(r['identity_lanes_equal'].values())
 assert r['orientation_counts']==({'same':r['shell_face_rows'],'reverse':0,'unmatched':0} if role=='O' else {'same':0,'reverse':r['shell_face_rows'],'unmatched':0})
 tests.append({'name':role+'_signed_match','status':'PASS'})
bad=Arrays(skin);bad['registered_I_source_face_rows']=skin['registered_I_source_face_rows'].copy();bad['registered_I_source_face_rows'][0]+=1
reject('wrong_historical_ancestry',lambda:orientation.check_asset('I',state['source_surfaces']['I'],bad,fat))
bad=Arrays(skin);bad['registered_I_source_face_ids']=skin['registered_I_source_face_ids'].copy();bad['registered_I_source_face_ids'][0]+=1
reject('wrong_anatomical_identity',lambda:orientation.check_asset('I',state['source_surfaces']['I'],bad,fat))
bad=Arrays(skin);bad['registered_I_source_faces']=skin['registered_I_source_faces'][:-1]
reject('missing_face',lambda:orientation.check_asset('I',state['source_surfaces']['I'],bad,fat))
bad=Arrays(skin);bad['registered_I_source_faces']=skin['registered_I_source_faces'].copy();bad['registered_I_source_faces'][1]=bad['registered_I_source_faces'][0]
reject('duplicate_face',lambda:orientation.check_asset('I',state['source_surfaces']['I'],bad,fat))
reject('wrong_asset_hash',lambda:orientation.check_asset('I',dict(state['source_surfaces']['I'],sha256='0'*64),skin,fat))
report={'schema':'numi.h706.current-433-signed-role-face-binding-controls.v1','status':'PASS','physical_admission':False,'native_steps':0,'controls':tests,'sources':{str(p):sha(p) for p in [Path(__file__),C/'typed-adapter/typed-current-schema.py',C/'typed-adapter/audit-role-orientation.py']}}
(C/'signed-face-controls-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print('negative_controls=PASS count='+str(len(tests)),flush=True)
# Extend the existing geometry-admission owner for this explicitly pinned joint quotient.
p=H/'current-native-1025-launch/current-native-1025-direct-o-006-source-geometry-scope/build_source_geometry_admission_001.py'
s=p.read_text();a=s.index('C=H/');b=s.index('def sha(',a)
s=s[:a]+"""C=Path("""+repr(str(C))+""");V=C/'volume-export';Q=C
P806=C/'proposal-exact-806/report.json';P808=C/'source-proof-808.json'
Q1020=Path("""+repr(str(D/'report.json'))+""");STATE=C/'verified-state.json'
SHELL=C/'shells/receipt.json';ORIENT=C/'role-orientation.json';SIGNED=C/'signed-face-controls-receipt.json'
CELL=C/'mass-allocation/stdout.log'
ROUTES=H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json'
BOUNDS=Path('/Users/numi/Work/skin-fat-native-integration-209/apps/NumiHumanWholeBodySkinReduction.hpp')
OUT=C/'source-geometry-admission-001.json'
SPEC=OUT.with_name('source-geometry-admission-spec-001.json');PINS=OUT.with_name('source-geometry-candidate-input-pins-001.json')
EXPECTED="""+repr({str(p):sha(p) for p in [C/'verified-state.json',C/'proposal-exact-806/report.json',C/'shells/receipt.json',C/'volume-export/receipt.json',C/'role-orientation.json',C/'mass-allocation/stdout.log',H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json']+[C/f'volume-export/roles/{r}-contact-thickness.csv' for r in 'OID']})+"""
ASSETS="""+repr({r:(str(D/f'candidate-{r}-typed-current.npz'),sha(D/f'candidate-{r}-typed-current.npz')) for r in 'OID'})+"""
CHILD="""+repr({n:sha(C/'volume-export/volume'/n) for n in ['mesh.node','mesh.ele','child.labels.csv']})+"\n\n"+s[b:]
s=s.replace('gate.get("status")=="PASS_ZERO_SKIN_FAT_QUALITY_806_808_SOURCE_ONLY"','gate.get("status")=="JOINT_BOUNDARY_VOLUME_PROPOSAL_NOT_PHYSICAL_ADMISSION"')
s=s.replace('gate.get("surface_crossings")==0 and gate.get("counts")=={"Skin":0,"Fat":0}','gate.get("native_quality_failures")==0 and gate.get("all_point_coordinates_unchanged") is True')
s=s.replace('state.get("status")=="PASS_ZERO_SKIN_FAT_SOURCE_QUALITY_MASS_PENDING"','state.get("status")=="PASS_SOURCE_ONLY_CONNECTED_BOUNDARY_VOLUME_QUALITY"')
with np.load(D/'candidate-I-typed-current.npz') as z: shared_faces=len(z['faces'])
s=s.replace('shared.get("source_faces")==236389','shared.get("source_faces")=='+str(shared_faces))
s=s.replace('q.get("source_face_rows_identity") is True','q.get("source_face_ancestry_exact") is True')
with np.load(C/'rich433/candidate-current-typed-volume.npz') as z: node_count=len(z['points_reference_world_m_f64']);cell_count=len(z['volume_tetrahedra'])
s=s.replace('==311768','=='+str(node_count)).replace('==1495294','=='+str(cell_count))
# Explicit limitation: triangle crossing checks are not solid-containment qualification.
s=s.replace('"checks":checks,"pair_counts":pair_counts,','"limitations":["Exact changed-face crossings checked against retained baseline; no new complete solid-containment claim","Native motion and performance remain unqualified"],"checks":checks,"pair_counts":pair_counts,')
out=C/'build_source_geometry_admission_001.py';out.write_text(s)
m=load(out,'joint64sourcegeometry');m.main()
