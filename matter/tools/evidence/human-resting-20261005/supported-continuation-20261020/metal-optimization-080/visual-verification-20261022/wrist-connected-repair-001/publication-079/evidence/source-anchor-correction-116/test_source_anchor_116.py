import importlib.util
from pathlib import Path
import numpy as np

SCRIPT=Path('/Users/n/Work/numi-human-connected-surface-069/digit-source-diagnosis-110/source-anchor-correction-116/source_anchor_batch_116.py')
spec=importlib.util.spec_from_file_location('source_anchor_116',SCRIPT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_direct_id_resolves_exact_source_vertex_not_other_candidate():
 rows=[{'detail_source_vertex_id':'7331'},{'detail_source_vertex_id':'7536'}]
 xyz=np.array([[17.1,0.,0.],[0.,0.,0.]])
 rec={'source_direct_id':'7536','source_face_v0':'7536','source_face_v1':'7331','source_face_v2':'7331','source_bary0':'1','source_bary1':'0','source_bary2':'0','world_x_m':'0','world_y_m':'0','world_z_m':'0'}
 a=m.detail_anchor(rec,{7331:0,7536:1},xyz,rows)
 assert a['kind']=='direct_source_vertex' and a['detail_row']==1
 assert np.array_equal(a['point_m'],xyz[1])
 assert np.linalg.norm(a['point_m']-xyz[0])==17.1

def test_virtual_anchor_is_exact_barycentric_surface_point():
 rows=[{'detail_source_vertex_id':str(i)} for i in range(3)]
 xyz=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
 rec={'source_direct_id':'4294967295','source_face_v0':'0','source_face_v1':'1','source_face_v2':'2','source_bary0':'0.25','source_bary1':'0.5','source_bary2':'0.25','world_x_m':'1','world_y_m':'0.5','world_z_m':'0'}
 a=m.detail_anchor(rec,{0:0,1:1,2:2},xyz,rows)
 assert a['kind']=='source_face_barycentric_virtual_point'
 assert np.allclose(a['point_m'],[1.,.5,0.],rtol=0,atol=0)
 assert abs(sum(a['barycentric'])-1)<1e-15

def test_invalid_non_sentinel_direct_id_fails_closed():
 rows=[{'detail_source_vertex_id':'7'}];xyz=np.zeros((1,3))
 rec={'source_direct_id':'9','source_face_v0':'7','source_face_v1':'7','source_face_v2':'7','source_bary0':'1','source_bary1':'0','source_bary2':'0','world_x_m':'0','world_y_m':'0','world_z_m':'0'}
 try:m.detail_anchor(rec,{7:0},xyz,rows)
 except ValueError as e:assert 'absent' in str(e)
 else:raise AssertionError('invalid direct ID silently fell back')
