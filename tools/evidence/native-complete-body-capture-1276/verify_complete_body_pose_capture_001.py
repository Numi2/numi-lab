from pathlib import Path
import sys,json,hashlib,importlib.util,inspect,numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";N=A/"twelve-surface-native-composition-001/baseline/native-run";B=A/"ten-surface-native-composition-001/baseline/native-run"
O=A/"twelve-surface-native-composition-001/complete-body-pose-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import common_atlas_skin_clearance as ca
H=R/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py"
sp=importlib.util.spec_from_file_location("prior",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=N.parent.parent/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T)
src=inspect.getsource(h.forward).replace("if w<=f(2e-6): continue","if w<=f(0): continue")
# Native tissue accumulation includes every strictly positive influence.
fp=O/"native_positive_weight_forward.py";fp.write_text(src);exec(compile(src,str(fp),"exec"),h.__dict__)
pins={str(p):sha(p) for p in [Path(__file__),H,fp,T,Path(ca.__file__)]}
out={"scope":"Additive all-body accepted geometry receipt verification. All source tissue body owners covered, registered subset unchanged, native entire body and respiratory state hashes matched to prior ten-surface run. Stratified actual-render checks for previously omitted Abdomen owner; not full trunk anatomy clearance.","pins":pins,"captures":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
needed={int(x["core"]) for x in t["bindings"]}
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 rp=N/"accepted-geometry"/f"step-{step}.receipt.json";oldp=B/"accepted-geometry"/rp.name;pack=rp.parent/f"step-{step}.mrvpack"
 for p in [rp,oldp,pack]:pins[str(p)]=sha(p)
 rc=json.loads(rp.read_text());old=json.loads(oldp.read_text());assert rc["physical_endpoint"]=="accepted" and rc["pack_file_sha256"]==pins[str(pack)]
 rows=rc["accepted_body_poses"];count=rc["accepted_body_pose_count"];assert len(rows)==count
 assert [x["body_index"] for x in rows]==list(range(count))
 poses={x["body_index"]:(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rows}
 assert needed<=set(poses)
 byid={x["body_index"]:x for x in rows};assert all(byid[x["body_index"]]==x for x in rc["accepted_registered_body_poses"])
 assert rc["accepted_registered_body_poses"]==old["accepted_registered_body_poses"]
 assert rc["accepted_body_state_sha256"]==old["accepted_body_state_sha256"]
 assert rc["accepted_respiration_state_sha256"]==old["accepted_respiration_state_sha256"]
 missing_old=sorted(needed-{x["body_index"] for x in old["accepted_registered_body_poses"]})
 positions,surfaces,_=ca._pack_surfaces(pack,{(51005,67),(51005,68)})
 cr={"step":step,"body_pose_count":count,"tissue_binding_owners":len(needed),"formerly_missing_tissue_body_owners":missing_old,"all_tissue_body_owners_covered":True,"registered_subset_exact":True,"full_body_state_hash_exact":True,"respiratory_state_hash_exact":True,"replay_samples":[]}
 for sid in [67,68]:
  row=h.row_data(t,sid);sf=np.asarray(surfaces[(51005,sid)]["faces"],int);delta=sf-row["faces"];assert np.all(delta==delta.flat[0])
  used=np.unique(row["faces"]);sample=used[np.unique(np.linspace(0,len(used)-1,128,dtype=int))]
  predicted=h.forward(row["positions"][sample],row["local"][sample],row["weights"][sample],row,t,poses)
  actual=positions[sample+int(delta.flat[0])];err=np.linalg.norm(predicted.astype(float)-actual.astype(float),axis=1)
  assert np.isfinite(predicted).all() and float(err.max())<1e-6,(step,sid,float(err.max()))
  cr["replay_samples"].append({"stable_id":sid,"sample_count":len(sample),"maximum_referenced_forward_error_m":float(err.max())})
 out["captures"].append(cr);save()
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()
print(json.dumps({"complete":True,"inputs_unchanged":out["inputs_unchanged"],"captures":len(out["captures"]),"body_count":out["captures"][0]["body_pose_count"],"formerly_missing":out["captures"][0]["formerly_missing_tissue_body_owners"]}),flush=True)
