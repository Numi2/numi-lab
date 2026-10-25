import hashlib,json
from pathlib import Path
import numpy as np
R=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-source-patch-cut-123")
O=R/"run-030-fine-parent-metadata-correction"
P134=R/"run-028-paired-ear-retriangulation/hybrid-O-surface-134.npz"
P133=R/"run-027-global-hybrid/hybrid-O-surface-133.npz"
CUT=R/"run-026-global-seed-closure-source-cut/merged-source-patch-cut.npz"
OUT=O/"hybrid-O-surface-135.npz"; REPORT=O/"metadata-correction-report.json"
PIN={P134:"41dfcbaea73d0385f0df50a091bb5dd39c1411b62b5d014805e29bb586b55ee0",P133:"cc11c68d58ec492922c3c86247197e8d83bbaf1fec9f55cc1a6fc3fedbad5273",CUT:"627bb87bcf81b5e84b3d68cb0c5666621b0979a214b34924be1a51ab8276e095"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True)
 if OUT.exists() or REPORT.exists(): raise FileExistsError("run-030 exists")
 for p,h in PIN.items():
  if sha(p)!=h: raise RuntimeError("SHA mismatch "+str(p))
 a=np.load(P133,allow_pickle=False); b=np.load(P134,allow_pickle=False); c=np.load(CUT,allow_pickle=False)
 n=len(b["source_O_faces"]); k=b["face_kind"]; rows=b["retriangulation_output_face_rows"].astype(int)
 orig=b["retriangulation_original_source_face_rows"].astype(int); sub=a["parent_fine_detail_face_row"].astype(int)
 if n!=17507 or not np.array_equal(np.sort(orig),[14642,14650]): raise RuntimeError("schema mismatch")
 if not np.array_equal(a["points_world_m"],b["points_world_m"]): raise RuntimeError("point geometry changed")
 changed=np.flatnonzero(np.any(a["source_O_faces"]!=b["source_O_faces"],axis=1))
 if not np.array_equal(changed,np.sort(rows)): raise RuntimeError("unexpected topology changes")
 if not np.array_equal(a["face_kind"],k): raise RuntimeError("kind changes")
 parent=c["fine_subface_parent_rows"].astype(int); src=b["parent_source_face_marker"].astype(int)
 fine=k==2; pair=np.zeros(n,bool); pair[rows]=True; regular=fine&~pair
 if not np.array_equal(parent[sub[regular]],src[regular]): raise RuntimeError("detail/subface mismatch")
 pair_sub=sub[rows]; pair_detail=parent[pair_sub]
 if not np.array_equal(np.sort(pair_detail),np.sort(orig)): raise RuntimeError("paired ancestry mismatch")
 z={name:np.asarray(b[name]).copy() for name in b.files}
 detail=np.full(n,-1,np.int32); detail[regular]=src[regular]
 subrow=np.full(n,-1,np.int32); subrow[regular]=sub[regular]
 z["parent_fine_detail_face_row"]=detail; z["parent_fine_subface_row"]=subrow
 z["parent_fine_detail_face_row_semantics"]=np.asarray("original NHSKIN face row; -1 for multi-source ancestry")
 z["parent_fine_subface_row_semantics"]=np.asarray("run-026 fine_subfaces row; -1 when not unique")
 ptr=np.zeros(n+1,np.int64); ent=[]
 for i in range(n):
  if regular[i]: ent.append(int(sub[i]))
  elif pair[i]: ent.extend(map(int,pair_sub))
  ptr[i+1]=len(ent)
 ent=np.asarray(ent,np.int32); z["face_fine_subface_ancestry_indptr"]=ptr; z["face_fine_subface_ancestry_rows"]=ent
 sp=b["face_source_ancestry_indptr"]; sr=b["face_source_ancestry_face_rows"]
 for i in rows:
  if not np.array_equal(np.sort(sr[sp[i]:sp[i+1]]),np.sort(orig)): raise RuntimeError("source ancestry CSR mismatch")
  if ptr[i+1]-ptr[i]!=2 or detail[i]!=-1 or subrow[i]!=-1: raise RuntimeError("replacement ancestry malformed")
  if not np.array_equal(np.sort(parent[ent[ptr[i]:ptr[i+1]]]),np.sort(orig)): raise RuntimeError("subface ancestry malformed")
 with OUT.open("xb") as f: np.savez_compressed(f,**z)
 q=np.load(OUT,allow_pickle=False)
 report={"status":"METADATA_CORRECTED_GEOMETRY_UNCHANGED","inputs":{str(p):h for p,h in PIN.items()},"output":str(OUT),"output_sha256":sha(OUT),
 "geometry_identical_to_134":{"points_world_m":bool(np.array_equal(q["points_world_m"],b["points_world_m"])),"source_O_faces":bool(np.array_equal(q["source_O_faces"],b["source_O_faces"]))},
 "face_count":n,"fine_faces":int(fine.sum()),"unique_detail_parent_faces":int(regular.sum()),"retriangulation_face_rows":rows.tolist(),
 "paired_detail_face_ancestry":orig.tolist(),"paired_subface_rows":pair_sub.tolist(),"paired_subface_detail_rows":pair_detail.tolist(),
 "source_ancestry_by_face":{str(int(i)):sr[sp[i]:sp[i+1]].astype(int).tolist() for i in rows},
 "fine_subface_ancestry_by_face":{str(int(i)):ent[ptr[i]:ptr[i+1]].astype(int).tolist() for i in rows},
 "semantics":"parent_fine_detail_face_row now names original NHSKIN face rows; parent_fine_subface_row separately names run-026 fine subfaces. Retriangulated faces carry both source and subface ancestry in CSR. No geometry/self/bone/mass/native admission implied."}
 with REPORT.open("x") as f: json.dump(report,f,indent=2,sort_keys=True); f.write("\n")
 print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
