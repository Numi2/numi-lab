#!/usr/bin/env python3
import hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import validate_charts as chart
V=Path(__file__).with_name("validate_charts.py")
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture():
 b=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.5,.25,.25]])
 f=np.array([[0,1,3],[1,2,3],[2,0,3]],np.int64)
 return {"points_world_m":b[:,:3],"source_O_faces":f,"refinement_face_parent_face_row":np.zeros(3,dtype=np.int64),"refinement_face_parent_barycentric":b[f],"source_prolongation_indptr":np.arange(5,dtype=np.int64),"source_prolongation_node_rows":np.arange(4,dtype=np.int64),"source_prolongation_weights":np.ones(4)}
def candidate(pin):
 b=np.eye(3);return {"points_world_m":b.copy(),"source_O_faces":np.array([[0,1,2]],np.int64),"refinement_face_parent_face_row":np.array([0],np.int64),"refinement_face_parent_barycentric":b[None],"source_prolongation_indptr":np.arange(4,dtype=np.int64),"source_prolongation_node_rows":np.arange(3,dtype=np.int64),"source_prolongation_weights":np.ones(3),"remesh806_baseline_sha256":np.array([pin]),"remesh806_old_to_new_node_rows":np.array([0,1,2,-1],np.int64),"remesh806_new_to_old_node_rows":np.array([0,1,2],np.int64),"remesh806_face_parent_indptr":np.array([0,3],np.int64),"remesh806_face_parent_face_rows":np.array([0,1,2],np.int64)}
def run(root,base,d,name,good):
 path=root/(name+".npz");out=root/(name+".json");np.savez(path,**d)
 r=subprocess.run([sys.executable,str(V),str(base),str(path),str(out),"--expected-baseline",sh(base),"--expected-candidate",sh(path)],capture_output=True,text=True)
 if (r.returncode==0)!=good:raise AssertionError((name,r.stdout,r.stderr))
 if good:
  j=json.loads(out.read_text()); assert j["altered_charts"]==1 and j["overlap_ancestry_pairs_checked"]==3 and j["deleted_nodes"]==1
def main():
 a=np.array([[0.,0.],[1.,0.],[0.,1.]])
 b=np.array([[.25,0.],[.75,0.],[.25,.5]])
 touch=np.array([[1.,0.],[2.,0.],[1.,1.]])
 assert abs(chart.area(chart.clip(a,b))-.125)<1e-13
 assert abs(chart.area(chart.clip(b,a))-.125)<1e-13
 assert chart.area(chart.clip(a,touch))<1e-13
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);base=root/"base.npz";f=fixture(); chain={p+s:np.array([0,1]) for p in ("prior_input_006__","prior_run028__") for s in ("source_prolongation_indptr","source_prolongation_node_rows","source_prolongation_weights")};f.update(chain);np.savez(base,**f);c=candidate(sh(base));c.update(chain);run(root,base,c,"valid",True)
  d=c.copy();d["remesh806_face_parent_indptr"]=np.array([0,2]);d["remesh806_face_parent_face_rows"]=np.array([0,1]);run(root,base,d,"omitted_parent",False)
  d=c.copy();d["refinement_face_parent_barycentric"]=c["refinement_face_parent_barycentric"].copy();d["refinement_face_parent_barycentric"][0,0]=[.98,.01,.01];run(root,base,d,"stale_material_chart",False)
  d=c.copy();d["source_prolongation_node_rows"]=np.array([0,2,1]);run(root,base,d,"changed_material_source",False)
  d=c.copy();d["remesh806_new_to_old_node_rows"]=np.array([0,1,3]);run(root,base,d,"noninverse_mapping",False)
  d=c.copy();d["remesh806_baseline_sha256"]=np.array(["0"*64]);run(root,base,d,"bad_pin",False)
  d=c.copy();d["remesh806_face_parent_face_rows"]=np.array([0,1,1]);run(root,base,d,"duplicate_parent",False)
  d=c.copy();d["prior_run028__source_prolongation_node_rows"]=np.array([1,0]);run(root,base,d,"changed_older_source",False)
 print("PASS: valid 3-to-1 patch and 7 rejected ancestry/map corruptions")
if __name__=="__main__":main()
