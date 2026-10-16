from pathlib import Path
import datetime,hashlib,json
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
R=Q/"checkout-publication032"
O=Q/"analysis/publication033-milestone-doc-review-001"
assert not O.exists()
O.mkdir()
doc1=R/"matter/examples/cardboard/solver-milestone-source198.md"
doc2=R/"matter/examples/cardboard/implementation-20261014.md"
a=doc1.read_bytes(); b=doc2.read_bytes()
t1=a.decode();t2=b.decode()
checks={
 "source198_milestone_mentions_binding": "publication-binding-reference033.json" in t1,
 "source198_milestone_says_all_six_axes_open": "All six overall axes remain open" in t1,
 "source198_milestone_excludes_active_native521": "fresh native521 coupon is excluded while active" in t1,
 "source198_milestone_limits_fixture_and_coupon": "does not calibrate corrugated board" in t1 and "No complete coupon" in t1,
 "source198_milestone_preserves_scalar_failure_masking": "before reaching the injected fault" in t1,
 "source198_milestone_preserves_cgls_nonconvergence": "without convergence" in t1,
 "implementation_ledger_includes_source198_open_status": "Source198" in t2 and "remain open" in t2
}
assert all(checks.values())
h=lambda x:hashlib.sha256(x).hexdigest()
report={
 "schema":"numi.cardboard.publication033-milestone-doc-review.v1",
 "verdict":"PASS",
 "documents":{
  str(doc1.relative_to(R)):{"bytes":len(a),"sha256":h(a)},
  str(doc2.relative_to(R)):{"bytes":len(b),"sha256":h(b)}
 },
 "scope_checks":checks,
 "physical_or_complete_coupon_qualification_claimed":False,
 "finished_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()
}
rb=(json.dumps(report,indent=2,sort_keys=True)+"\n").encode()
(O/"report.json").write_bytes(rb)
script=Path(__file__)
(O/"audit.py").write_bytes(script.read_bytes())
manifest={
 "schema":"numi.cardboard.publication033-milestone-doc-review.manifest.v1",
 "inputs":{"binding_sha256":h((R/"matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference033.json").read_bytes()),
           "summary_sha256":h((R/"matter/examples/cardboard/evidence/native-implementation-20261014/summary.json").read_bytes()),
           "publication033_independent_review_manifest_sha256":h((Q/"analysis/publication033-independent-review-006/manifest.json").read_bytes())},
 "outputs":{n:{"bytes":len(p.read_bytes()),"sha256":h(p.read_bytes())} for n,p in [( "audit.py",O/"audit.py"),("report.json",O/"report.json")]}
}
mb=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode();(O/"manifest.json").write_bytes(mb)
print(json.dumps({"report":report,"manifest_sha256":h(mb)},indent=2))

