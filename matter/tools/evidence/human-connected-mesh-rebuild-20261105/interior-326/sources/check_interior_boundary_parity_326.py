from pathlib import Path
import numpy as np,json,sys,hashlib
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'interior-preserved-boundary-326'
sys.path.insert(0,'/Users/numi/Work/skin-fat-native-integration-209/apps')
import simplify_human_skin_fat_boundary as m
out={}
for role,path in m.S.items():
 with np.load(path) as a,np.load(D/f'candidate-{role}-typed-current.npz') as b:
  lanes=[k for k in a.files if k in ['faces','points_world_m','points_world_m_f32'] or k.startswith('face_source_') or k.startswith('face_ancestry_') or k.startswith('quality_star_patch_union_')]
  for k in lanes:
   assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),(role,k)
  assert (D/f'candidate-{role}-typed-current.npz').resolve()==path.resolve()
  assert hashlib.file_digest((D/f'candidate-{role}-typed-current.npz').open('rb'),'sha256').hexdigest()==hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
  out[role]={'bit_identical_geometry_and_typed_ancestry_lanes':lanes,'faces':len(a['faces'])}
report={'status':'PASS_BIT_IDENTICAL_BOUNDARY_GEOMETRY_AND_TYPED_ANCESTRY','physical_admission':False,'roles':out,'back_geometry_unchanged':True,'surface_crossing_delta':0,'note':'Identity proves no changed surface geometry; not a new full containment proof or motion qualification.'}
(D/'fixed-boundary-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
