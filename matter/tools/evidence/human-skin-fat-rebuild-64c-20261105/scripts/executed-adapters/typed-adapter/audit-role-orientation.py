from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
TASK=ROOT/"current-native-1024-root/current-433-1044"
LAUNCH=ROOT/"current-native-1024-root/current-433-1043/launch-inputs.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def can(tri):
 x=tuple(map(int,tri));return min(x[i:]+x[:i] for i in range(3))
def orient(a,b):
 if can(a)==can(b):return 1
 if can(a)==can((int(b[0]),int(b[2]),int(b[1]))):return -1
 return 0
def check_asset(role, pin, skin, fat):
 path=Path(pin["path"]);digest=pin["sha256"]
 if sha(path)!=digest:raise ValueError(f"{role} current typed SHA mismatch")
 shell=skin if role in ("O","I") else fat
 prefix=f"registered_{role}_"
 srcv=np.asarray(shell[prefix+"source_vertex_rows"],dtype=np.int64)
 shellv=np.asarray(shell[prefix+"shell_node_rows"],dtype=np.int64)
 srctri_shell=np.asarray(shell[prefix+"source_faces"],dtype=np.int64)
 shellrows=np.asarray(shell[prefix+"shell_face_rows"],dtype=np.int64)
 shelltri_stored=np.asarray(shell[prefix+"shell_faces"],dtype=np.int64) if prefix+"shell_faces" in shell.files else np.asarray(shell["faces"],dtype=np.int64)[shellrows]
 if len(np.unique(srcv))!=len(srcv) or len(np.unique(shellv))!=len(shellv):raise ValueError(f"{role} shell vertex map is not injective")
 node_to_source={int(sn):int(vr) for vr,sn in zip(srcv,shellv)}
 with np.load(path,allow_pickle=False) as a:
  afaces=np.asarray(a["faces"],dtype=np.int64)
  identity={k:np.asarray(a[k],dtype=np.int64) for k in ("face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id") if k in a.files}
  if not np.array_equal(srctri_shell,afaces):raise ValueError(f"{role} 1022 registered source triangles are not exact typed row order")
  if role=="I":
   source_face_rows=np.asarray(shell["registered_I_source_face_rows"],dtype=np.int64)
   expected_ancestry_rows=np.asarray(a["source_input_face_rows"],dtype=np.int64) if "source_input_face_rows" in a.files else np.arange(len(afaces),dtype=np.int64)
   if source_face_rows.shape!=(len(afaces),) or not np.array_equal(source_face_rows,expected_ancestry_rows):raise ValueError("I 1022 face ancestry differs from pinned typed source")
   if not np.array_equal(np.asarray(shell["registered_I_source_face_ids"],dtype=np.int64),identity["face_source_face_id"]):raise ValueError("I shell face ID row map differs from typed source")
  else:
   source_face_rows=np.arange(len(afaces),dtype=np.int64)
  mapped_shell=np.asarray([[node_to_source[int(n)] for n in tri] for tri in shelltri_stored],dtype=np.int64)
  signs=np.asarray([orient(a,b) for a,b in zip(afaces,mapped_shell)],dtype=np.int8)
  fields_ok={}
  for key,vals in identity.items():
   shell_key=key
   if shell_key in shell.files:
    fields_ok[key]=bool(np.array_equal(vals, np.asarray(shell[shell_key])[shellrows]))
  if not identity or set(fields_ok)!=set(identity):
   raise ValueError(f"{role} shell is missing one or more typed face identity lanes")
  shell_role=np.asarray(shell["face_source_role"]).astype(str)[shellrows]
  shell_marker=np.asarray(shell["face_marker"],dtype=np.int64)[shellrows]
  shell_face_rows_actual=np.asarray(shell["faces"],dtype=np.int64)[shellrows]
  if not np.array_equal(shell_face_rows_actual,shelltri_stored):raise ValueError(f"{role} stored shell triangles differ from actual shell face rows")
  return {"role":role,"asset":{"path":str(path),"sha256":digest,"faces":len(afaces),"points":len(a["points_world_m_f32"])},"shell_face_rows":len(shellrows),"source_triangle_row_order_exact":True,"source_face_rows_identity":bool(np.array_equal(source_face_rows,np.arange(len(afaces)))),"source_face_ancestry_exact":True,"identity_lanes_equal":fields_ok,"shell_face_roles":{str(x):int(np.count_nonzero(shell_role==x)) for x in np.unique(shell_role)},"shell_face_markers":{str(x):int(np.count_nonzero(shell_marker==x)) for x in np.unique(shell_marker)},"orientation_counts":{"same":int(np.count_nonzero(signs==1)),"reverse":int(np.count_nonzero(signs==-1)),"unmatched":int(np.count_nonzero(signs==0))},"first_source_triangle":afaces[0].tolist(),"first_shell_triangle_as_source_rows":mapped_shell[0].tolist(),"first_face_identity":{k:int(v[0]) for k,v in identity.items()}}
def main():
 import argparse
 ap=argparse.ArgumentParser(description="Audit current typed role winding against a specifically pinned fresh 1022 shell")
 ap.add_argument("--state",type=Path,required=True)
 ap.add_argument("--receipt",type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args()
 state_path=a.state.resolve();receipt_path=a.receipt.resolve();out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 state=json.loads(state_path.read_text());receipt=json.loads(receipt_path.read_text())
 if state.get("physical_admission") is not False or state.get("native_steps",0)!=0:
  raise ValueError("orientation audit requires source-only current state")
 if receipt.get("schema")!="numi.h706.current-paired-shell-1022.v1" or receipt.get("status")!="PASS_CURRENT_938V2_PAIRED_SHELLS_SOURCE_ONLY":
  raise ValueError("fresh paired 1022 shell receipt is not a source-only pass")
 if receipt.get("physical_admission") is not False or receipt.get("native_steps")!=0:
  raise ValueError("fresh 1022 shell receipt overclaims physical/native evidence")
 state_pin={"path":str(state_path),"sha256":sha(state_path)}
 if receipt.get("current_state")!={"path":str(state_path),"sha256":state_pin["sha256"],"size_bytes":state_path.stat().st_size}:
  raise ValueError("fresh 1022 shell receipt does not bind the supplied current state")
 source=state.get("source_surfaces",{})
 if set(source)!={"O","I","D"}:raise ValueError("current state must bind exactly O/I/D source surfaces")
 if receipt.get("inputs",{}).get("source_surfaces")!=source:
  raise ValueError("1022 shell source surfaces differ from current verified state")
 for role,item in source.items():
  p=Path(item["path"])
  if not p.is_file() or sha(p)!=item["sha256"]:raise ValueError("current "+role+" source pin missing or stale")
 outputs=receipt.get("outputs",{})
 skin_item=outputs.get("skin-shell.npz");fat_item=outputs.get("fat-shell.npz")
 if not isinstance(skin_item,dict) or not isinstance(fat_item,dict):raise ValueError("fresh shell receipt lacks Skin/Fat shell outputs")
 skin_path=Path(skin_item["path"]);fat_path=Path(fat_item["path"])
 for label,item,path in (("Skin shell",skin_item,skin_path),("Fat shell",fat_item,fat_path)):
  if not path.is_file() or sha(path)!=item.get("sha256"):raise ValueError(label+" output pin is missing or stale")
 roles={}
 with np.load(skin_path,allow_pickle=False) as skin, np.load(fat_path,allow_pickle=False) as fat:
  for role in ("O","I","D"):roles[role]=check_asset(role,source[role],skin,fat)
 orient=receipt.get("assembly",{}).get("orientation",{})
 expected={"O":1,"I":-1,"D":-1}
 expected_roles={}
 for role in ("O","I","D"):
  n=int(roles[role]["shell_face_rows"])
  expected_roles[role]={"same":n,"reverse":0,"unmatched":0} if role=="O" else {"same":0,"reverse":n,"unmatched":0}
 ok=(all(roles[r]["orientation_counts"]==expected_roles[r] for r in expected)
  and all(x["identity_lanes_equal"] and all(x["identity_lanes_equal"].values()) for x in roles.values())
  and all(roles[r]["source_triangle_row_order_exact"] and roles[r]["source_face_ancestry_exact"] for r in expected)
  and orient.get("O_flipped_to_wall") is False and orient.get("I_skin_flipped_to_wall") is True
  and orient.get("D_flipped_to_wall") is True and orient.get("fat_global_positive_volume_flip") is False
  and orient.get("skin_global_positive_volume_flip") is False)
 report={"schema":"numi.h706.1022-role-surface-orientation-audit.v1",
  "status":"PASS_EXACT_1022_SIGNED_ROLE_CONVENTIONS" if ok else "FAIL_ROLE_ORIENTATION_OR_IDENTITY",
  "validator":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__))},
  "current_state":state_pin,"receipt":{"path":str(receipt_path),"sha256":sha(receipt_path)},
  "shell_assets":{"skin":{"path":str(skin_path),"sha256":sha(skin_path)},"fat":{"path":str(fat_path),"sha256":sha(fat_path)}},
  "1022_orientation_contract":orient,"source_to_material_outward_sign":expected if ok else None,
  "roles":roles,"physical_admission":False,"native_steps":0}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps(report,sort_keys=True))
 if not ok:raise SystemExit(2)
if __name__=="__main__":main()
