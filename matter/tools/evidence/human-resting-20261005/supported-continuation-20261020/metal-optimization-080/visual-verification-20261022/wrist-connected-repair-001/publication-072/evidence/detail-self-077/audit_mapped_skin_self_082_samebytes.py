from pathlib import Path
import hashlib, json, signal, time, gc, resource, sys
import numpy as np
from scipy import sparse

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
WORK = Path("/Users/n/Work/numi-human-connected-surface-069")
OUT = WORK / "self-intersection-077" / "run-004-082-080"
OUT.mkdir(parents=True, exist_ok=False)
INPUT = ROOT / "E/connected-volume-refinement-037/run-006/candidate.npz"
CANDIDATE = WORK / "direct-authoring-082/repair-run-embedded-082/iteration-001/candidate.npz"
TRANSFER_DIR = ROOT / "accepted-transfer-row-export-075/run-001/source-preflight"
TRANSFER_ROWS = TRANSFER_DIR / "accepted-transfer-rows.csv"
COARSE_REFERENCE = TRANSFER_DIR / "accepted-coarse-reference.csv"
COARSE_FACES = TRANSFER_DIR / "accepted-coarse-faces.csv"
DETAIL_PAYLOAD = ROOT / "bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"
ADAPTER = ROOT / "E/accepted-transfer-row-adapter-075/load_transfer_075.py"
SURFACE_OWNER = WORK / "canonical/matter/tools/human_skin_fat_embedded_surface.py"
PREDICATE = ROOT / "collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
AUDIT_DEPS = ROOT / "incremental-exact-audit-owner-253/src"
EXPECTED = {
 "baseline_037": "52c81dbe323eab2240bc315f7b7bde605a201fa8e928206f27000de7e5d7a492",
 "transfer_rows": "71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14",
 "coarse_reference": "9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2",
 "coarse_faces": "7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2",
 "detail_payload": "7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb",
 "adapter": "6f1c8887ab33d75bf5dd6241ec7a4a4ae8bc21d2d7573a8ba36b417f303c1117",
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def rss_max():
 return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
def write_json(path, obj):
 Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
def save_pairs(path, pairs):
 np.save(path,np.asarray(pairs,dtype=np.int64).reshape(-1,2),allow_pickle=False)
pins={"baseline_037":str(INPUT),"candidate_082_samebytes_080":str(CANDIDATE),"transfer_rows":str(TRANSFER_ROWS),
 "coarse_reference":str(COARSE_REFERENCE),"coarse_faces":str(COARSE_FACES),
 "detail_payload":str(DETAIL_PAYLOAD),"adapter":str(ADAPTER),
 "mapped_surface_owner":str(SURFACE_OWNER),"exact_predicate":str(PREDICATE)}
for k,p in pins.items():
 if k in EXPECTED:
  actual=sha(p)
  if actual!=EXPECTED[k]: raise RuntimeError(f"{k} pin mismatch {actual}")
pins["hashes"]={k:sha(p) for k,p in pins.items()}
pins["audit_dependency_hashes"]={name:sha(AUDIT_DEPS/"numilab_human"/name) for name in ("__init__.py","cardiac_cavity_geometry.py","model.py","physiology.py")}
pins["candidate_082_samebytes_080_sha256"]=sha(CANDIDATE)
EXPECTED_CANDIDATE_SHA="cd717a9ca78d92b02ae9d30c1a3da6602993d229524d48dc62d0722092519c30"
if pins["candidate_082_samebytes_080_sha256"]!=EXPECTED_CANDIDATE_SHA: raise RuntimeError("082/080 same-byte candidate SHA mismatch")
PARENT_RUNNER=WORK/"self-intersection-077/audit_mapped_skin_self_077_v3.py"
EXPECTED_PARENT_RUNNER_SHA="b42d9f77fb3c6463806ed3458277862cbc6c49f2714556c8afa0cd4d98a0db0b"
if sha(PARENT_RUNNER)!=EXPECTED_PARENT_RUNNER_SHA: raise RuntimeError("pinned 077 runner changed")
pins["parent_runner"]=str(PARENT_RUNNER)
pins["parent_runner_sha256"]=sha(PARENT_RUNNER)
pins["scope"]="mapped detailed NHSKIN self-intersection exact audit; physical/native admission not evaluated"
write_json(OUT/"input-pins.json",pins)
stage="setup";start=time.monotonic()
def timeout_handler(signum,frame): raise TimeoutError("five-minute audit budget reached")
signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(285)
report={"status":"running","scope":pins["scope"],"stage":stage,"started_unix":time.time(),"baseline":None,"candidate":None}
write_json(OUT/"progress.json",report)
try:
 sys.path.insert(0,str(ADAPTER.parent));sys.path.insert(0,str(SURFACE_OWNER.parent));sys.path.insert(0,str(AUDIT_DEPS))
 import importlib.util
 from load_transfer_075 import load_transfer075
 from human_skin_fat_embedded_surface import MappedSkinSurface
 spec=importlib.util.spec_from_file_location("numilab_human._cardiac_cavity_intersections_315",PREDICATE)
 audit315=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit315;spec.loader.exec_module(audit315)
 float32_surface_records=audit315.float32_surface_records
 _audit_self_pairs_touching_faces=audit315._audit_self_pairs_touching_faces
 transfer_pins={k:EXPECTED[k] for k in ("transfer_rows","coarse_reference","coarse_faces","detail_payload")}
 W075,detail_origin,coarse_transfer_origin,faces=load_transfer075(TRANSFER_ROWS,COARSE_REFERENCE,COARSE_FACES,DETAIL_PAYLOAD,expected_sha256=transfer_pins)
 z=np.load(INPUT,allow_pickle=False);O=z["current_points_f32"]
 if O.dtype!=np.float32 or O.shape!=(9320,3):raise RuntimeError(f"bad 037 current points {O.shape} {O.dtype}")
 cz=np.load(CANDIDATE,allow_pickle=False);Y=cz["points_world_f32"]
 if Y.dtype!=np.float32 or Y.shape!=O.shape:raise RuntimeError(f"bad 076 points {Y.shape} {Y.dtype}")
 if not np.array_equal(coarse_transfer_origin,O[:W075.shape[1]]):raise RuntimeError("075 coarse origin is not bitwise 037 mechanics prefix")
 if W075.shape!=(61950,2411) or detail_origin.shape!=(61950,3) or faces.shape!=(108987,3):raise RuntimeError("unexpected pinned 075 mapped domain")
 W=sparse.hstack([W075,sparse.csr_matrix((len(detail_origin),len(O)-W075.shape[1]),dtype=np.float64)],format="csr")
 mapped=MappedSkinSurface(O.copy(),detail_origin,faces,W)
 baseline=mapped.positions(O)
 if not np.array_equal(baseline,detail_origin):raise RuntimeError("mapped baseline does not reproduce exact detail origin")
 candidate=mapped.positions(Y)
 if candidate.dtype!=np.float32 or candidate.shape!=detail_origin.shape or not np.isfinite(candidate).all():raise RuntimeError("mapped candidate is not finite Float32 detail geometry")
 movement=np.linalg.norm(candidate.astype(np.float64)-baseline.astype(np.float64),axis=1)
 report["mapping"]={"detail_vertices":int(len(detail_origin)),"detail_faces":int(len(faces)),"transfer_shape":list(W.shape),
  "transfer_nnz":int(W.nnz),"active_detail_rows":int(np.count_nonzero(np.diff(W.indptr))),
  "inactive_detail_rows":int(np.count_nonzero(np.diff(W.indptr)==0)),"baseline_bitwise_matches_detail_origin":True,
  "max_detail_displacement_mm":float(movement.max()*1000),"median_detail_displacement_mm":float(np.median(movement)*1000),
  "p99_detail_displacement_mm":float(np.quantile(movement,.99)*1000)}
 stage="baseline_records";report["stage"]=stage;write_json(OUT/"progress.json",report);print("STAGE baseline_records",flush=True)
 baseline_records,_=float32_surface_records(baseline,faces,source_seam_groups=())
 stage="baseline_exact_self_audit";report["stage"]=stage;write_json(OUT/"progress.json",report);print("STAGE baseline_exact_self_audit",flush=True)
 t=time.monotonic();base_result=_audit_self_pairs_touching_faces(baseline_records,set(range(len(faces))));base_elapsed=time.monotonic()-t
 base_pairs=base_result["triangle_pairs"];save_pairs(OUT/"baseline_triangle_pairs.npy",base_pairs)
 report["baseline"]={"status":"complete","faces":len(faces),"intersection_pairs":base_result["count"],
  "aabb_candidate_pairs":base_result["aabb_candidate_pairs"],"allowed_shared_vertex_or_edge_pairs":base_result["allowed_shared_vertex_or_edge_pairs"],
  "complete_surface_audit":base_result["complete_surface_audit"],"elapsed_seconds":base_elapsed,"peak_rss_bytes":rss_max()}
 report["stage"]="baseline_complete";write_json(OUT/"progress.json",report);print("BASELINE_COMPLETE "+json.dumps(report["baseline"]),flush=True)
 del baseline_records,base_result,baseline;gc.collect()
 if time.monotonic()-start>140:
  report["status"]="baseline_only_budget_stop";report["stage"]="candidate_skipped_budget";report["elapsed_seconds"]=time.monotonic()-start
  report["peak_rss_bytes"]=rss_max();write_json(OUT/"report.json",report);raise SystemExit(0)
 stage="candidate_records";report["stage"]=stage;write_json(OUT/"progress.json",report);print("STAGE candidate_records",flush=True)
 candidate_records,_=float32_surface_records(candidate,faces,source_seam_groups=())
 stage="candidate_exact_self_audit";report["stage"]=stage;write_json(OUT/"progress.json",report);print("STAGE candidate_exact_self_audit",flush=True)
 t=time.monotonic();candidate_result=_audit_self_pairs_touching_faces(candidate_records,set(range(len(faces))));candidate_elapsed=time.monotonic()-t
 candidate_pairs=candidate_result["triangle_pairs"];save_pairs(OUT/"candidate_triangle_pairs.npy",candidate_pairs)
 base_set=set(map(tuple,base_pairs));cand_set=set(map(tuple,candidate_pairs));new_pairs=sorted(cand_set-base_set);resolved=sorted(base_set-cand_set)
 save_pairs(OUT/"new_triangle_pairs.npy",new_pairs);save_pairs(OUT/"resolved_triangle_pairs.npy",resolved)
 report["candidate"]={"status":"complete","faces":len(faces),"intersection_pairs":candidate_result["count"],
  "aabb_candidate_pairs":candidate_result["aabb_candidate_pairs"],"allowed_shared_vertex_or_edge_pairs":candidate_result["allowed_shared_vertex_or_edge_pairs"],
  "complete_surface_audit":candidate_result["complete_surface_audit"],"elapsed_seconds":candidate_elapsed,"peak_rss_bytes":rss_max()}
 report["pair_comparison"]={"new_pairs":len(new_pairs),"resolved_pairs":len(resolved),"baseline_pairs":len(base_pairs),"candidate_pairs":len(candidate_pairs),
  "classification":"exact owner315 predicate; only intersections confined to true common indexed vertex/edge are allowed; no adjacency exemption",
  "new_pair_sample":new_pairs[:100],"resolved_pair_sample":resolved[:100]}
 report["status"]="complete";report["stage"]="complete";report["elapsed_seconds"]=time.monotonic()-start;report["peak_rss_bytes"]=rss_max()
 write_json(OUT/"report.json",report);np.savez_compressed(OUT/"surface-positions.npz",baseline_f32=detail_origin,candidate_f32=candidate)
 file_hashes={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="manifest.json"}
 write_json(OUT/"manifest.json",{"status":report["status"],"files":file_hashes,"input_pins":pins["hashes"]})
 print("AUDIT_COMPLETE "+json.dumps({"baseline_pairs":len(base_pairs),"candidate_pairs":len(candidate_pairs),"new_pairs":len(new_pairs),
  "resolved_pairs":len(resolved),"baseline_s":base_elapsed,"candidate_s":candidate_elapsed,"elapsed_s":report["elapsed_seconds"],
  "peak_rss":report["peak_rss_bytes"]}),flush=True)
except SystemExit:raise
except BaseException as exc:
 report["status"]="incomplete";report["stage"]=stage;report["error"]=repr(exc);report["elapsed_seconds"]=time.monotonic()-start;report["peak_rss_bytes"]=rss_max()
 write_json(OUT/"report.json",report);print("AUDIT_INCOMPLETE "+json.dumps(report),flush=True)
