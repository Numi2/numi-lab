"""Construction-only damping witnesses; never a geometry admission gate."""
import numpy as np
def regressed_point_faces(before,after,faces,node_count):
 if int(before.get("inside_node_bank_pairs",-1))!=0:
  raise ValueError("point-regression damping requires zero-inside reference")
 witnesses=after.get("inside_witnesses")
 if not isinstance(witnesses,list) or len(witnesses)!=int(after.get("inside_node_bank_pairs",-1)) or not witnesses:
  raise ValueError("complete nonempty point witness table required")
 rows=[]
 for w in witnesses:
  if w.get("classification")!="inside":raise ValueError("non-inside witness")
  i=int(w["output_node"])
  if not 0<=i<node_count:raise ValueError("point witness outside node domain")
  rows.append(i)
 rows=np.unique(np.asarray(rows,dtype=np.int64))
 incident=np.flatnonzero(np.isin(faces,rows).any(axis=1))
 if len(np.unique(faces[incident][np.isin(faces[incident],rows)]))!=len(rows):raise ValueError("unreferenced witness")
 return rows,incident
