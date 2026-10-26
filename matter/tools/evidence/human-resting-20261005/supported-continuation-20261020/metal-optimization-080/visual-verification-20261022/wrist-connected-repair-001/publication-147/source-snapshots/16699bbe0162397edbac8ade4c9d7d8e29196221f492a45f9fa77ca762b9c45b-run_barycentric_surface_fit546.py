"""Existing connected LP with full construction coordinates and surface-only goals.

This changes offline construction controls/objectives, not the native model.
Inside-depth monotonicity is a routing heuristic and is omitted; every outside
point-depth row, all self/bone guards, source/area/edge/anchor constraints remain.
Exact final geometry and all185-bank checks are still mandatory.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys,numpy as np
from scipy import sparse
from surface_objective_546 import retain_surface_rows546, select_outside_depth_rows546
D=Path('/Users/numi/Work/numi-human-connected-surface-069');ENTRY=D/'D/D/component-lp-decomposition-490/run_matched_component_fit_490.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 frozen_extra={Path('/Users/numi/Work/numi-human-connected-surface-069/D/contained-clearance-545/component_lp_active545.py'):'37c8a7e22f240be1a669e0f29712b0dfb068861fd89c9b6b2a8955c8ffdf39ed',Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/surface_objective_546.py'):'dd6f70afab9cb51f3658d8f4fc9d65a13a6f70517e536a9e81c4c2a098750516'}
 assert all(sha(k)==v for k,v in frozen_extra.items()),"frozen adapter/helper changed"
 assert sha(ENTRY)=='7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693'
 s=importlib.util.spec_from_file_location('fit490_surface546',ENTRY);entry=importlib.util.module_from_spec(s);s.loader.exec_module(entry);load=entry._load_module;receipt={}
 def load490(path):
  r=load(path)
  if Path(path).resolve()!=entry.RUNNER_485.resolve():return r
  old=r.load
  def load485(name,path):
   m=old(name,path)
   if name=='component_lp_decomposition_485_for_run':
    extra=Path("/Users/numi/Work/numi-human-connected-surface-069/D/contained-clearance-545/component_lp_active545.py");sp=importlib.util.spec_from_file_location('component_active545',extra);active=importlib.util.module_from_spec(sp);sp.loader.exec_module(active)
    m.solve_by_components=active.solve_by_components;receipt['active_component_adapter_source_sha256']=sha(extra)
    return m
   if name!='matched458_for_485':return m
   install=m.install_objective
   def install545(owner,objective,crossing):
    bary_build=owner.build_clearance_rows;result=install(owner,objective,crossing);basis=owner.build_immediate_parent_basis;build=owner.build_clearance_rows;depth=owner.build_output_node_direction_rows
    def full_basis(base,support,anchors):
     B,parent,br=basis(base,support,anchors);scope=np.asarray(br['output_scope_rows'],np.int64);free=np.flatnonzero(~np.isin(scope,anchors))
     Q=sparse.csr_matrix((np.ones(len(free)),(free,np.arange(len(free)))),shape=(len(scope),len(free)))
     br.update(control_parent_rows=scope[free].tolist(),matrix_shape=list(Q.shape),matrix_nnz=Q.nnz,semantics='independent emitted-node construction coordinates on the existing expanded scope; source ancestry and native mechanical coordinates unchanged',construction_full_output_coordinates=True)
     receipt['construction_coordinate_nodes']=len(free)
     return Q,scope[free],br
    def surface_rows(*args,**kwargs):
     plane_C,plane_h,plane_w,plane_meta=build(*args,**kwargs)
     receipt['plane_objective_rows_diagnostic_only']=len(plane_h)
     C,h,w,meta,proof=retain_surface_rows546(bary_build,*args,**kwargs)
     proof.pop('all_458_surface_soft_rows_retained',None)
     proof['all_408_barycentric_surface_soft_rows_retained']=True
     proof['objective_scope']='all matched 408 barycentric face targets; original normalized weights retained'
     proof['actual_objective']='original 408 barycentric targets at certified all185 union-exit endpoints; 458 nonpositive-corner hard guards retained'
     proof['active_soft_objective']='original 408 barycentric face targets'
     proof['458_positive_plane_rows_active']=False
     proof['458_positive_plane_rows_role']='diagnostic counts only; excluded from the LP soft objective'
     proof['458_nonpositive_corner_hard_rows_active']=True
     proof['458_nonpositive_corner_hard_rows_role']='active hard bounds appended by the retained 458 barrier-row combiner'
     receipt.update(proof)
     return C,h,w,meta
    def outside_rows(node_count,support,node_rows,normals,lower,typed_ids=None):
     nr,nn,lo,ids,proof=select_outside_depth_rows546(node_rows,normals,lower,typed_ids)
     receipt.update(proof)
     return depth(node_count,support,nr,nn,lo,ids)
    owner.build_immediate_parent_basis=full_basis;owner.build_clearance_rows=surface_rows;owner.build_output_node_direction_rows=outside_rows
    return result
   m.install_objective=install545;return m
  r.load=load485;return r
 entry._load_module=load490
 out=Path(sys.argv[sys.argv.index('--output')+1]);selfsha=sha(__file__);entry.main();assert sha(__file__)==selfsha;assert all(sha(k)==v for k,v in frozen_extra.items())
 p=out/'report.json';j=json.loads(p.read_text())
 plane=j['fixed_bone_outward_plane_objective_458']
 plane['active_soft_objective_replaced_by']='original408_barycentric_surface546'
 plane['active_soft_objective']=False
 plane['positive_bone_plane_target_rows_role']='diagnostic count only; these rows were excluded from the active LP soft objective'
 plane['positive_bone_plane_target_rows_active']=False
 plane['nonpositive_corner_hard_rows_active']=True
 plane['nonpositive_corner_hard_rows_role']='active hard bounds appended through the 458 barrier-row combiner'
 j['full_coordinate_barycentric_surface546']={**receipt,'source_sha256':selfsha,'construction_only':True,'native_admission':False,'accepted_physical_steps':0}
 p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
