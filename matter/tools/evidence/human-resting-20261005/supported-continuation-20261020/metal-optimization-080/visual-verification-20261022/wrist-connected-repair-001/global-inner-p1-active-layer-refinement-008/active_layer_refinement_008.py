"""Conforming local P1 boundary-layer support using the existing 403 owner."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
def owner403():
 p=E/'global-inner-p1-edge-support-403/refine_internal_o_edges_403.py'
 spec=importlib.util.spec_from_file_location('support_refine_403',p)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def select_internal_outer_free_edges(tetrahedra,physical_outer_node_ids,shell_faces,failed_tet_ids,node_count):
 """Select O/free chords only; true exterior edges are never subdivided."""
 T=np.asarray(tetrahedra);O=np.asarray(physical_outer_node_ids);S=np.asarray(shell_faces);F=np.asarray(failed_tet_ids)
 for v,shape,name in ((T,4,'tetrahedra'),(S,3,'shell_faces')):
  if v.ndim!=2 or v.shape[1]!=shape or not np.issubdtype(v.dtype,np.integer) or np.any(v<0) or np.any(v>=node_count):
   raise ValueError('invalid '+name)
 if O.ndim!=1 or not np.issubdtype(O.dtype,np.integer) or np.any(O<0) or np.any(O>=node_count) or len(np.unique(O))!=len(O):
  raise ValueError('invalid physical O IDs')
 if F.ndim!=1 or (F.size and not np.issubdtype(F.dtype,np.integer)) or np.any(F<0) or np.any(F>=len(T)):
  raise ValueError('failed tetra rows exceed current domain')
 fixed=np.zeros(node_count,bool);fixed[O]=True
 boundary={tuple(sorted((int(f[i]),int(f[j])))) for f in S for i,j in ((0,1),(0,2),(1,2))}
 pairs=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3));selected=set()
 for ti in np.unique(F).astype(np.int64):
  for i,j in pairs:
   aa,bb=map(int,T[ti,[i,j]]);edge=(min(aa,bb),max(aa,bb))
   if fixed[aa]!=fixed[bb] and edge not in boundary:selected.add(edge)
 return sorted(selected)

