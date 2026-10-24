from pathlib import Path
import hashlib, json, os, shutil, subprocess

R = Path("/Users/n/numi-human-aortic-contact-opt-20261019")
D = R / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
PUB_REL = "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/publication-055"
PUB = R / PUB_REL
PARENT = "f3c4acd72cd8363fa77b465509a051cf3578a72f"
ROOTS = ["connected-skin-envelope-fit-014","connected-skin-ray-envelope-016","connected-skin-safe-flow-017","connected-skin-exact-self-audit-018","current-detailed-skin-crossing-audit-018"]
SUPPORT = [
"collision-free-seam-records-owner-315/run_tests.py",
"collision-free-seam-records-owner-315/tests/test_cardiac_cavity_exact_plane_filter.py",
"collision-free-seam-records-owner-315/tests/test_cardiac_cavity_intersections.py",
"collision-free-seam-records-owner-315/tests/test_float32_source_seam_records.py",
"collision-free-seam-records-owner-315/tests.log",
"collision-free-seam-records-owner-315/tests-available.log",
"collision-free-seam-records-owner-315/qualification-receipt.json",
"collision-free-seam-records-owner-315/actual-native-pose-coordinate-parity.json",
"registered-pose-frame-helper-194/tools/rebind_human_skin_reference.py",
"registered-pose-frame-helper-194/tests/test_human_skin_reference_rebind.py",
"registered-pose-frame-helper-194/test-results.log",
"registered-pose-frame-helper-194/replay_native_registration.py",
"registered-pose-frame-helper-194/native-registration-parity.json",
"skin-repair-lexicographic-lp-276/tools/human_skin_repair_lp.py",
"skin-repair-lexicographic-lp-276/tests/test_human_skin_repair_lp.py",
"skin-repair-lexicographic-lp-276/tests.log",
"global-inner-p1-full-domain-clipping-009/exact-O-pair-011/audit_exact_pairs_011.py",
"global-inner-p1-full-domain-clipping-009/exact-O-pair-011/test_exact_pairs_011.py",
"global-inner-p1-full-domain-clipping-009/exact-O-pair-011/tests.log",
]
ALIASES = {
str(D/"connected-skin-envelope-fit-014/finite_witness_fit_014.py"): {
"4c6753ece020104afa6f55956ebd25f7571bf34e09fff8f33f1e7389476bebd7":"connected-skin-envelope-fit-014/finite_witness_fit_014-run005-executed.py",
"38f240a792a92fc6c6337e167053ab1fd362208107ec2cb02cef86b6bd8d5fd7":"connected-skin-envelope-fit-014/finite_witness_fit_014.py"},
str(D/"connected-skin-safe-flow-017/safe_connected_fit_017.py"): {
"6b1dd4581d97b97c4749934efc6339a5078224f30e4c9baa88c088ad2c280175":"connected-skin-safe-flow-017/safe_connected_fit_017-run001-executed.py",
"b90310933500710e118f5d3990b778a5e6b1bd452cb09d91ed797034a9c9dba9":"connected-skin-safe-flow-017/safe_connected_fit_017.py"},
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def run(args,env=None):
 return subprocess.run(args,cwd=R,env=env,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def clean_copytree(src,dst):
 shutil.copytree(src,dst,ignore=shutil.ignore_patterns("__pycache__","*.pyc",".DS_Store"))
def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); tmp.replace(path)
def main():
 if PUB.exists():
  if (PUB/"publication-manifest-055.json").exists():
   raise SystemExit("completed publication-055 already exists; refusing overwrite")
  shutil.rmtree(PUB)
 PUB.mkdir(parents=True)
 for rel in ROOTS:
  src=D/rel
  if not src.is_dir(): raise SystemExit("missing evidence directory: "+str(src))
  clean_copytree(src,PUB/rel)
 deps=PUB/"dependencies"
 for rel in SUPPORT:
  src=D/rel
  if not src.is_file(): raise SystemExit("missing dependency/test artifact: "+str(src))
  dst=deps/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
 shutil.copy2(Path(__file__),PUB/"prepare_publication_055.py")

 note="""# Publication 055 evidence correction

This is an evidence-only bundle for the hand/forearm connected-skin experiments. It changes no runtime source, asset, or accepted geometry. All proposals in this bundle remain unadmitted.

## Results and interpretation

- The 013 source-refinement baseline completed the full exact self-surface audit with 0 exact self-pairs across 38,104 faces.
- 014 run-005 is a finite-witness construction only. It retained 519 moved nodes, up to 43.3013 mm displacement, and a worst emitted constraint slack of -32.8395 mm. Its 39 nonpositive old/new normal cosines are a comparison diagnostic, not by themselves proof of a fold. The candidate was rejected.
- 016 run-001 moved 2,340 nodes by up to 24.9928 mm. The reported 76 nonpositive normal cosines are likewise not a fold proof. The exact predicate found 302 added self-intersection pairs, so the proposal is rejected.
- 017 run-001 retained positive tetrahedron determinants in its bounded proposal but stopped because no safe connected step remained; it was not an exact-contact or native acceptance.
- 017 run-002 has 0 added exact self-pairs against 013, but 78 nonpositive tetrahedra (minimum normalized determinant -0.1182); reject.
- 017 run-003 reduced target residual but produced 198 added exact self-pairs and 2,652 nonpositive tetrahedra (minimum normalized determinant -0.9337); reject. Its 47 reference normal deviations over 90 degrees are recorded only as diagnostics.
- The 018 exact self audit compares each candidate to the same complete 013 surface. 016 adds 302 exact pairs; 017-002 adds 0; 017-003 adds 198. Exact pairs are collision evidence; cosine-only normal differences are not.
- Detailed-skin audit run-001 is retained as a superseded hybrid: it combined identity-004 CSV positions with source-294 topology, so it is not an exact owner-replayed source-294 result. Run-002 is the corrected owner-194 Float32 replay of the pinned source-294 payload at identity-004 body poses. It finds 3,070 transverse skin/bank triangle intersections over the complete 108,987-face detailed skin against 9,964 selected facets in 67 banks. The selected tissue-bank side is incomplete, this is a preparation replay rather than a native capture, and native/physical admission remain false.

The 014 run-005 report references a later-overwritten source filename. Its exact executed bytes are preserved as finite_witness_fit_014-run005-executed.py and are mapped by the manifest. Similarly, 017 run-001 as-run source is preserved as safe_connected_fit_017-run001-executed.py. Original reports and logs remain byte-for-byte unchanged.

## Validation records

The retained unit logs report 3 passing tests for 014, 3 for 016, 3 for 017 incremental fitting, 3 for the current detailed-skin parser, 5 for the lexicographic LP helper, 28 for the registered-pose helper, and 55 for the exact intersection/seam suite when its one unavailable external-source test is excluded. The older exact-suite log also remains: 49 tests pass and one test errors because external anatomy source partof_element_parts.txt is unavailable. No test result here qualifies the geometry as native or physical.

## Remaining gates

No candidate is admitted. The 018 bank subset does not cover all registered tissue banks; no complete hand/fat transfer, all-pose attachment/topology/volume gate, native rebuild/replay, or physical acceptance is claimed. The results are bounded evidence for repair planning only.
"""
 (PUB/"evidence-correction-055.md").write_text(note)
 test_rows=[
 {"suite":"connected-skin-envelope-fit-014","log":"connected-skin-envelope-fit-014/tests.log","reported":"3 passed"},
 {"suite":"connected-skin-ray-envelope-016","log":"connected-skin-ray-envelope-016/tests.log","reported":"3 passed"},
 {"suite":"connected-skin-safe-flow-017","log":"connected-skin-safe-flow-017/incremental-tests.log","reported":"3 passed"},
 {"suite":"current-detailed-skin-crossing-audit-018","log":"current-detailed-skin-crossing-audit-018/tests.log","reported":"3 passed"},
 {"suite":"skin-repair-lexicographic-lp-276","log":"dependencies/skin-repair-lexicographic-lp-276/tests.log","reported":"5 passed"},
 {"suite":"registered-pose-frame-helper-194","log":"dependencies/registered-pose-frame-helper-194/test-results.log","reported":"28 passed"},
 {"suite":"collision-free-seam-records-owner-315","log":"dependencies/collision-free-seam-records-owner-315/tests-available.log","reported":"55 passed; one external-source test excluded"},
 {"suite":"collision-free-seam-records-owner-315-older-run","log":"dependencies/collision-free-seam-records-owner-315/tests.log","reported":"49 passed, 1 error: unavailable partof_element_parts.txt"},
 ]
 for row in test_rows:
  p=PUB/row["log"]
  if not p.is_file(): raise SystemExit("missing test log "+str(p))
  row["sha256"]=sha(p)
 atomic_json(PUB/"test-results-055.json",{"status":"RETAINED_EXISTING_TEST_LOGS_NOT_RERUN","results":test_rows})

 refs={}
 for root_name in ROOTS:
  for rp in (PUB/root_name).rglob("*.json"):
   try: obj=json.loads(rp.read_text())
   except Exception: continue
   def walk(x):
    if isinstance(x,dict):
     m=x.get("input_sha256")
     if isinstance(m,dict):
      for path,digest in m.items(): refs[(str(path),str(digest))]=str(rp.relative_to(PUB))
     for v in x.values(): walk(v)
    elif isinstance(x,list):
     for v in x: walk(v)
   walk(obj)
 input_rows=[]; failures=[]
 for (reported,expected),report_rel in sorted(refs.items()):
  alias_rel=ALIASES.get(reported,{}).get(expected)
  resolved=(D/alias_rel) if alias_rel else Path(reported)
  actual=sha(resolved) if resolved.is_file() else None
  status="MATCH" if actual==expected else ("MISSING" if actual is None else "MISMATCH")
  row={"reported_path":reported,"expected_sha256":expected,"verification_path":str(resolved),
       "actual_sha256":actual,"status":status,"referenced_by_report":report_rel}
  if alias_rel: row["as_run_source_snapshot"]="matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/"+alias_rel
  input_rows.append(row)
  if status!="MATCH": failures.append(row)
 if failures: raise SystemExit("input pin validation failed: "+json.dumps(failures,indent=2))
 atomic_json(PUB/"report-input-pins-055.json",{"status":"ALL_REPORT_DECLARED_INPUT_HASHES_VERIFIED","unique_path_digest_pairs":len(input_rows),"executed_source_aliases":ALIASES,"pins":input_rows})

 source_pins=[
 ("collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py","f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d","inherited at parent"),
 ("registered-pose-frame-helper-194/tools/rebind_human_skin_reference.py","be36923a62ebae0ecd475b57d5a413f424712c4fa129c23e6fe4620479869693","copied into dependencies"),
 ("skin-repair-lexicographic-lp-276/tools/human_skin_repair_lp.py","6e4a3eb2b6355b082b3bc1b69e5a0325b5aa2abe61f4b75c5e12b9becc446c8c","copied into dependencies"),
 ("global-inner-p1-full-domain-clipping-009/exact-O-pair-011/audit_exact_pairs_011.py",None,"copied into dependencies"),
 ]
 source_check=[]
 for rel,expected,where in source_pins:
  path=(D/rel) if where=="inherited at parent" else (deps/rel)
  if not path.is_file(): raise SystemExit("missing source pin "+str(path))
  actual=sha(path)
  if expected is not None and actual!=expected: raise SystemExit("source hash mismatch "+rel+" "+actual)
  source_check.append({"source":rel,"sha256":actual,"location":where})
 report2=json.loads((PUB/"current-detailed-skin-crossing-audit-018/run-002/report.json").read_text())
 external=[]
 for p,h in report2["input_sha256"].items():
  if not Path(p).is_relative_to(D):
   if not Path(p).is_file() or sha(p)!=h: raise SystemExit("external source input changed: "+p)
   external.append((p,h))

 payload=[]
 for p in sorted(PUB.rglob("*")):
  if not p.is_file() or p.name in {"payload-files-055.json","publication-manifest-055.json"}: continue
  payload.append({"path":str(p.relative_to(PUB)),"bytes":p.stat().st_size,"sha256":sha(p)})
 filelist={"schema":"publication-055-payload-files-v1","root_relative_to":"R","repository_root":str(R),
           "publication_path":PUB_REL,"files":payload,"file_count":len(payload),"total_bytes":sum(x["bytes"] for x in payload)}
 atomic_json(PUB/"payload-files-055.json",filelist)
 atomic_json(PUB/"publication-manifest-055.json",{
  "schema":"wrist-connected-repair-publication-055-v1","status":"PREPARED_FOR_REVIEW_UNPUBLISHED",
  "repository":"numi-human-aortic-contact-opt-20261019","repository_root":str(R),"target_parent_commit":PARENT,
  "current_checkout_head_at_prepare":run(["git","rev-parse","HEAD"]).stdout.strip(),
  "origin_main_at_prepare":run(["git","ls-remote","origin","refs/heads/main"]).stdout.split()[0],
  "publication_path_from_repository_root":PUB_REL,
  "scope":["014 constrained connected-skin fit and failed iterations","016 ray-envelope proposal","017 incremental fits and retained exact self audit","018 complete exact self audit","018 detailed-skin vs selected-bank exact crossing audit, corrected owner194 replay"],
  "excluded":["active015","mechanical019","runtime/production source changes","all physical/native admission claims"],
  "payload_file_count":len(payload),"payload_total_bytes":sum(x["bytes"] for x in payload),
  "payload_filelist_sha256":sha(PUB/"payload-files-055.json"),
  "report_input_pin_file":"report-input-pins-055.json","report_input_pin_file_sha256":sha(PUB/"report-input-pins-055.json"),
  "source_pins":source_check,"external_run002_inputs":[{"path":p,"sha256":h} for p,h in external],
  "provenance_corrections":[
   "014 run-005 report source path resolves to preserved finite_witness_fit_014-run005-executed.py; report remains unchanged.",
   "017 run-001 report source path resolves to preserved safe_connected_fit_017-run001-executed.py; report remains unchanged.",
   "Detailed audit 018 run-001 is a hybrid identity004 CSV-position/source294-topology result and is explicitly superseded for source294 replay by run-002.",
   "Run-002 uses owner194 Float32 replay of source294 at identity004 poses; it is a preparation replay, not a native capture."
  ],
  "created_files_are_evidence_only":True,"prepare_script_sha256":sha(PUB/"prepare_publication_055.py")
 })
 head=run(["git","rev-parse","HEAD"]).stdout.strip();remote=run(["git","ls-remote","origin","refs/heads/main"]).stdout.split()[0]
 if remote!=PARENT: raise SystemExit("origin/main moved from requested parent")
 index_path=Path("/tmp/publication-055-alt.index")
 if index_path.exists(): index_path.unlink()
 env=os.environ.copy();env["GIT_INDEX_FILE"]=str(index_path)
 run(["git","read-tree",PARENT],env=env);run(["git","add","-f","--",PUB_REL],env=env)
 tree=run(["git","write-tree"],env=env).stdout.strip();run(["git","diff","--check",PARENT,tree],env=env)
 commit=run(["git","commit-tree",tree,"-p",PARENT,"-m","Add scoped hand repair evidence publication 055"]).stdout.strip()
 if run(["git","cat-file","-t",commit]).stdout.strip()!="commit": raise SystemExit("prepared object is not commit")
 if run(["git","rev-list","--parents","-n","1",commit]).stdout.split()[1]!=PARENT: raise SystemExit("prepared commit has wrong parent")
 if run(["git","rev-parse","HEAD"]).stdout.strip()!=head: raise SystemExit("normal checkout HEAD changed")
 if run(["git","ls-remote","origin","refs/heads/main"]).stdout.split()[0]!=PARENT: raise SystemExit("origin/main changed")
 final={"parent":PARENT,"tree":tree,"prepared_commit":commit,"normal_head_unchanged":True,"origin_main_unchanged":True,
 "alternate_index":str(index_path),"files":len(payload),"bytes":sum(x["bytes"] for x in payload),
 "manifest_sha256":sha(PUB/"publication-manifest-055.json"),"filelist_sha256":sha(PUB/"payload-files-055.json")}
 Path("/tmp/publication-055-prepared-result.json").write_text(json.dumps(final,indent=2)+"\n")
 print(json.dumps(final,indent=2))
if __name__=="__main__": main()
