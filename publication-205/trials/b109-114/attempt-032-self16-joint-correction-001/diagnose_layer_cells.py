from pathlib import Path
import importlib.util,json,sys,csv
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-032-self16-joint-correction-001")
W=D.parent.parent
BASE=W/"combined-d071-b109-b61-001/candidate.npz";CAND=D/"candidate-segment-partial-alpha-0p0625.npz";LYR=W/"combined-d071-b109-b61-001/skin-fat-layers-541.npz"
TRANSFER=W/"b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py"
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def main():
 with np.load(BASE,allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
 with np.load(CAND,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
 with np.load(LYR,allow_pickle=False) as z:l={k:z[k].copy() for k in z.files}
 F=np.asarray(b["source_O_faces"],np.int64);LF=F[:,[0,2,1]];O=np.asarray(c["points_world_m"],np.float64)
 bo=np.asarray(l["outer_points_f64"],np.float64);bi=np.asarray(l["inner_points_f64"],np.float64);bd=np.asarray(l["deep_points_f64"],np.float64)
 Ic=O+(bi-bo);Dc=Ic+(bd-bi);tr=mod("tr",TRANSFER)
 baseS=tr.face_flags(bo,bi,LF,l["skin_thickness_transferred_m"]);carryS=tr.face_flags(O,Ic,LF,l["skin_thickness_transferred_m"])
 baseF=tr.face_flags(bi,bd,LF,l["fat_thickness_transferred_m"]);carryF=tr.face_flags(Ic,Dc,LF,l["fat_thickness_transferred_m"])
 rows=[]
 for layer,faces,new,old in [("skin",[34588,34708,34741],carryS,baseS),("fat",[35427],carryF,baseF)]:
  for f in faces:
   rows.append({"layer":layer,"face":f,"nodes":LF[f].tolist(),"baseline_bad":bool(old["authored_bad"][f]),"candidate_bad":bool(new["authored_bad"][f]),"baseline_nonpositive":old["nonpositive"][f].tolist(),"candidate_nonpositive":new["nonpositive"][f].tolist(),"baseline_conditioned":old["conditioned"][f].tolist(),"candidate_conditioned":new["conditioned"][f].tolist(),"baseline_volume_m3":old["volumes"][f].tolist(),"candidate_volume_m3":new["volumes"][f].tolist(),"baseline_ndet":old["normalized_det"][f].tolist(),"candidate_ndet":new["normalized_det"][f].tolist()})
 out={"targets":rows,"baseline_skin_bad_faces":int(baseS["authored_bad"].sum()),"baseline_fat_bad_faces":int(baseF["authored_bad"].sum()),"baseline_skin_nonpositive":int(baseS["nonpositive"].sum()),"baseline_fat_nonpositive":int(baseF["nonpositive"].sum()),"baseline_skin_conditioned":int(baseS["conditioned"].sum()),"baseline_fat_conditioned":int(baseF["conditioned"].sum())}
 p=D/"layer-target-cell-diagnostic.json";p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"report":str(p),"targets":rows},indent=2))
if __name__=="__main__":main()

