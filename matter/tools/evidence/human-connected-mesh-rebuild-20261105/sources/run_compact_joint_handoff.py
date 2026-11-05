from pathlib import Path
import ast,hashlib,importlib.util,json,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
D=Path(sys.argv[1]).resolve();C=D/'compiler'
def pin(p):
 p=Path(p)
 with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':sha}
# Execute the original, pinned compiler. Replace only its face-incidence storage.
driver=H/'current-role-p1-support-938-1025-active-source-map/current_fine_volume_to_217_938_818c_849d_active_rows.py'
assert pin(driver)['sha256']=='bce0755224226eb89c8e545e0a66f421c6e71aa7b0a479e00a8c39367e884e8f'
helper=Path('/Users/numi/Work/skin-fat-native-integration-209/apps/numi_human_tet_face_incidence.py')
sys.path.insert(0,str(helper.parent))
from numi_human_tet_face_incidence import CompactTetFaceMap
spec=importlib.util.spec_from_file_location('current_compact_storage_handoff',driver);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_import=m.import_owner
def import_owner():
 owner=original_import() # original source, capacity and identity admission
 owner._tetrahedron_boundary_faces=CompactTetFaceMap
 return owner
m.import_owner=import_owner
original_execute=m.execute
def execute(*args,**kwargs):
 result=original_execute(*args,**kwargs)
 p=Path(args[8])/'adapter-receipt.json'
 receipt=json.loads(p.read_text())
 extension={'scope':'INITIALIZATION_ONLY_COMPACT_FACE_INCIDENCE_STORAGE','original_owner':pin(m.OWNER_PATH),'wrapper':pin(__file__),'face_incidence':pin(helper),'semantics':'All face occurrences, materials, orientation and nonmanifold rejection retained; original compiler assertions executed'}
 receipt['bookkeeping_extension']=extension
 receipt['inputs']['compact_incidence']=pin(helper)
 receipt['inputs']['compact_handoff_wrapper']=pin(__file__)
 p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 result['bookkeeping_extension']=extension;result['receipt']=pin(p)
 return result
m.execute=execute
tree=ast.parse((W/'compile_joint64.py').read_text())
call=next(n.value for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='run' and n.value.args[0].value=='handoff938')
args=eval(compile(ast.Expression(call.args[1]),'<existing handoff command>','eval'),{'H':H,'D':D,'C':C,'current':C/'volume-prefix/verified-state.json'})
sys.argv=[str(driver),*map(str,args[1:])]
raise SystemExit(m.main())
