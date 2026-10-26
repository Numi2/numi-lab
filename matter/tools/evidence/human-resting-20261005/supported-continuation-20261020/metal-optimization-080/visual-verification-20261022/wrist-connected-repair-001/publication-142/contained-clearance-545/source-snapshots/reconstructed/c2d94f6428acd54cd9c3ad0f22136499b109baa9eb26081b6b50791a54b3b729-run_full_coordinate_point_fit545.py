"""Existing connected LP with full construction coordinates and point-only goals.

This changes offline construction controls/objectives, not the native model.
Inside-depth monotonicity is a routing heuristic and is omitted; every outside
point-depth row, all self/bone guards, source/area/edge/anchor constraints remain.
Exact final geometry and all185-bank checks are still mandatory.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys,numpy as np
from scipy import sparse
D=Path('/Users/numi/Work/numi-human-connected-surface-069');ENTRY=D/'D/D/component-lp-decomposition-490/run_matched_component_fit_490.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert sha(ENTRY)=='7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693'
 s=importlib.util.spec_from_file_location('fit490_fullpoint545',ENTRY);entry=importlib.util.module_from_spec(s);s.loader.exec_module(entry);load=entry._load_module;receipt={}
 def load490(path):
  r=load(path)
  if Path(path).resolve()!=entry.RUNNER_485.resolve():return r
  old=r.load
  def load485(name,path):
   m=old(name,path)
   if name!='matched458_for_485':return m
   install=m.install_objective
   def install545(owner,objective,crossing):
    result=install(owner,objective,crossing);basis=owner.build_immediate_parent_basis;build=owner.build_clearance_rows;depth=owner.build_output_node_direction_rows
    def full_basis(base,support,anchors):
     B,parent,br=basis(base,support,anchors);scope=np.asarray(br['output_scope_rows'],np.int64);free=np.flatnonzero(~np.isin(scope,anchors))
     Q=sparse.csr_matrix((np.ones(len(free)),(free,np.arange(len(free)))),shape=(len(scope),len(free)))
     br.update(control_parent_rows=scope[free].tolist(),matrix_shape=list(Q.shape),matrix_nnz=Q.nnz,semantics='independent emitted-node construction coordinates on the existing expanded scope; source ancestry and native mechanical coordinates unchanged',construction_full_output_coordinates=True)
     receipt['construction_coordinate_nodes']=len(free)
     return Q,scope[free],br
    def point_rows(*args,**kwargs):
     C,h,w,meta=build(*args,**kwargs);keep=np.array([m['kind']=='point'for m in meta]);assert keep.any()
     receipt.update(point_rows=int(keep.sum()),omitted_surface_soft_rows=int((~keep).sum()))
     new=[dict(m,normalized_weight=1/int(keep.sum()))for m,k in zip(meta,keep)if k]
     return C[keep],h[keep],np.full(int(keep.sum()),1/int(keep.sum())),new
    def outside_rows(node_count,support,node_rows,normals,lower,typed_ids=None):
     keep=np.asarray(lower)<0;receipt['omitted_inside_monotonicity_rows']=int((~keep).sum());receipt['retained_outside_depth_rows']=int(keep.sum())
     return depth(node_count,support,np.asarray(node_rows)[keep],np.asarray(normals)[keep],np.asarray(lower)[keep],None if typed_ids is None else np.asarray(typed_ids)[keep])
    owner.build_immediate_parent_basis=full_basis;owner.build_clearance_rows=point_rows;owner.build_output_node_direction_rows=outside_rows
    return result
   m.install_objective=install545;return m
  r.load=load485;return r
 entry._load_module=load490
 out=Path(sys.argv[sys.argv.index('--output')+1]);selfsha=sha(__file__);entry.main();assert sha(__file__)==selfsha
 p=out/'report.json';j=json.loads(p.read_text());j['full_coordinate_point545']={**receipt,'source_sha256':selfsha,'construction_only':True,'native_admission':False,'accepted_physical_steps':0};p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
