from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
W=S/"per-bank-workers-20261027"
D=W/"b109-114/attempt-031-alpha-scaled-progress-screen-001"
BASE=W/"combined-d071-b109-b61-001/candidate.npz"
BASE_SHA="806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756"
AUD159=W/"combined-d071-b109-b61-001/audit-159/report.json"
AUDIT=ROOT/"audit_skin_bone_surfaces_159.py"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
BODY=109;BONE=114
CANDS=[(0.0625,D/"candidate-segment-partial-alpha-0p0625.npz")]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mod(name,p):
 spec=importlib.util.spec_from_file_location(name,p);x=importlib.util.module_from_spec(spec);sys.modules[name]=x;spec.loader.exec_module(x);return x
def csr(z,pk,vk,n):
 p=np.asarray(z[pk],np.int64);v=np.asarray(z[vk],np.int64)
 if p.shape!=(n+1,) or p[0]!=0 or p[-1]!=len(v) or np.any(p[1:]<p[:-1]):raise ValueError("invalid ancestry CSR "+pk)
 return [list(map(int,v[p[i]:p[i+1]])) for i in range(n)]
def map_pairs(raw,parents):
 return {(int(p),int(b)) for f,b in raw for p in parents[int(f)]}
def main():
 t0=time.monotonic()
 if sha(BASE)!=BASE_SHA:raise ValueError("pinned 806 baseline SHA mismatch")
 a=json.loads(AUD159.read_text())
 if a.get("candidate_sha256")!=BASE_SHA or a.get("complete") is not True or int(a.get("bank_count",-1))!=185:raise ValueError("baseline audit159 pin/completeness mismatch")
 bref=next(v for v in a["banks"] if (int(v["body"]),int(v["stable_bone_id"]))==(BODY,BONE))
 expected={(int(f),int(b)) for f,b in (bref.get("result") or {}).get("triangle_pairs",[])}
 audit=mod("target_screen_audit159",AUDIT);helper=mod("target_screen_helper",HELPER);exact=audit.exact_module()
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
 BT=np.asarray(T[(ids[:,0]==BODY)&(ids[:,1]==BONE)],np.float64)*1000.
 if len(BT)!=392:raise ValueError("unexpected target bone triangle row count")
 bone_rows=np.arange(len(BT),dtype=np.int64)
 result={"schema":"numi.human.b109-114.target-identity-backtrack-screen.v1","baseline_sha256":BASE_SHA,"audit159_sha256":sha(AUD159),"target_bank":[BODY,BONE],"target_bone_triangle_row_count":len(BT),"candidate_results":[],"whole_185_bank_audit_run":False,"construction_only":True,"physical_steps":0}
 with np.load(BASE,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
 x0=np.asarray(base["points_world_m"],np.float32);F=np.asarray(base["source_O_faces"],np.int64)
 srcparents=csr(base,"face_source_ancestry_indptr","face_source_ancestry_face_rows",len(F))
 immediate=csr(base,"refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows",len(F))
 local0=((x0.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY])*1000.
 raw0={(int(f),int(b)) for f,b in (audit.intersect_bank(local0,F,BT,bone_rows,exact).get("result") or {}).get("triangle_pairs",[])}
 if raw0!=expected:raise ValueError(f"zero-motion exact control mismatch saved={len(expected)} local={len(raw0)} saved_only={sorted(expected-raw0)[:12]} local_only={sorted(raw0-expected)[:12]}")
 src0=map_pairs(raw0,srcparents);imm0=map_pairs(raw0,immediate)
 result["baseline_control"]={"raw_pair_count":len(raw0),"source_parent_pair_count":len(src0),"immediate_parent_pair_count":len(imm0),"saved_audit159_match":True}
 for alpha,path in CANDS:
  if not path.exists():raise ValueError("missing backtrack candidate "+str(path))
  with np.load(path,allow_pickle=False) as z:trial={k:z[k].copy() for k in z.files}
  x1=np.asarray(trial["points_world_m"],np.float32);F1=np.asarray(trial["source_O_faces"],np.int64)
  if x1.dtype!=np.float32 or F1.shape!=F.shape or not np.array_equal(F1,F):raise ValueError("candidate is not same-topology Float32 position patch")
  changed=np.flatnonzero(np.any(x1.view(np.uint32)!=x0.view(np.uint32),axis=1))
  affected=np.flatnonzero(np.any(np.isin(F,changed),axis=1))
  local1=((x1.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY])*1000.
  out=audit.intersect_bank(local1,F1,BT,bone_rows,exact)
  raw1={(int(f),int(b)) for f,b in (out.get("result") or {}).get("triangle_pairs",[])}
  src1=map_pairs(raw1,srcparents);imm1=map_pairs(raw1,immediate)
  result["candidate_results"].append({"alpha":alpha,"candidate_path":str(path),"candidate_sha256":sha(path),"changed_node_count":int(len(changed)),"changed_face_count":int(len(affected)),"changed_face_rows":affected.astype(int).tolist(),"baseline_raw_pair_count":len(raw0),"trial_raw_pair_count":len(raw1),"raw_removed":[list(v) for v in sorted(raw0-raw1)],"raw_added":[list(v) for v in sorted(raw1-raw0)],"source_parent_baseline_pair_count":len(src0),"source_parent_trial_pair_count":len(src1),"source_parent_removed":[list(v) for v in sorted(src0-src1)],"source_parent_added":[list(v) for v in sorted(src1-src0)],"immediate_parent_baseline_pair_count":len(imm0),"immediate_parent_trial_pair_count":len(imm1),"immediate_parent_removed":[list(v) for v in sorted(imm0-imm1)],"immediate_parent_added":[list(v) for v in sorted(imm1-imm0)],"aabb_candidate_pairs":int((out.get("result") or {}).get("aabb_candidate_pairs",-1)),"strict_source_parent_reduction":bool(src0-src1),"no_new_target_raw_pairs":not bool(raw1-raw0),"no_new_source_parent_pairs":not bool(src1-src0)})
 result["elapsed_s"]=time.monotonic()-t0
 outp=D/"target-identity-backtrack-screen.json"
 outp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":"TARGET_SCREEN_COMPLETE","report":str(outp),"report_sha256":sha(outp),"trials":[{k:r[k] for k in ("alpha","candidate_sha256","changed_node_count","changed_face_count","baseline_raw_pair_count","trial_raw_pair_count","raw_removed","raw_added","source_parent_baseline_pair_count","source_parent_trial_pair_count","source_parent_removed","source_parent_added","strict_source_parent_reduction","no_new_target_raw_pairs","no_new_source_parent_pairs")} for r in result["candidate_results"]]},indent=2,allow_nan=False))
if __name__=="__main__":main()
