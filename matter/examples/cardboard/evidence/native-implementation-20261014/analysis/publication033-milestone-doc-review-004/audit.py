from pathlib import Path
import datetime,hashlib,json
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
R=Q/"checkout-publication032"
O=Q/"analysis/publication033-milestone-doc-review-004"
assert not O.exists()
O.mkdir()
d1=R/"matter/examples/cardboard/solver-milestone-source198.md"
d2=R/"matter/examples/cardboard/implementation-20261014.md"
e=R/"matter/examples/cardboard/evidence/native-implementation-20261014/analysis/publication033-milestone-draft-004/evidence-index.json"
a=d1.read_bytes(); b=d2.read_bytes(); idxb=e.read_bytes()
t1=a.decode(); t2=b.decode(); idx=json.loads(idxb)
c523=next(c for c in idx["native_cases"] if c["case"]=="523")
checks={
 "milestone_links_binding":"publication-binding-reference033.json" in t1,
 "milestone_says_all_six_open":"All six overall axes remain open" in t1,
 "milestone_excludes_active521":"fresh native521 coupon is excluded while active" in t1,
 "milestone_limits_fixture":"does not calibrate corrugated board" in t1,
 "milestone_preserves_525_527_masking":"before reaching the injected fault" in t1,
 "milestone_says_523_hits_caps":"complete caps32/64" in t1,
 "milestone_says_523_private_only":"zero physical FEM transactions" in t1,
 "index_523_says_caps_not_convergence":"without convergence" in c523["result"],
 "implementation_ledger_retains_open_status":"Source198" in t2 and "remain open" in t2
}
assert all(checks.values())
H=lambda x:hashlib.sha256(x).hexdigest()
report={
 "schema":"numi.cardboard.publication033-milestone-doc-review.v1",
 "verdict":"PASS_WITH_WORDING_NOTE",
 "documents":{
  str(d1.relative_to(R)):{"bytes":len(a),"sha256":H(a)},
  str(d2.relative_to(R)):{"bytes":len(b),"sha256":H(b)}
 },
 "evidence_index_sha256":H(idxb),
 "scope_checks":checks,
 "wording_note":"The milestone accurately says positive damping paths complete 32/64 iteration caps and are private checks with zero physical transactions. The evidence index says the damped paths reach iteration caps without convergence; the milestone does not literally repeat that phrase, but it does not claim convergence.",
 "physical_or_complete_coupon_qualification_claimed":False,
 "finished_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()
}
rb=(json.dumps(report,indent=2,sort_keys=True)+"\n").encode()
(O/"report.json").write_bytes(rb)
(O/"audit.py").write_bytes(Path(__file__).read_bytes())
manifest={
 "schema":"numi.cardboard.publication033-milestone-doc-review.manifest.v1",
 "inputs":{
  "binding_sha256":H((R/"matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference033.json").read_bytes()),
  "summary_sha256":H((R/"matter/examples/cardboard/evidence/native-implementation-20261014/summary.json").read_bytes()),
  "evidence_index_sha256":H(idxb),
  "independent_review_manifest_sha256":H((Q/"analysis/publication033-independent-review-006/manifest.json").read_bytes())
 },
 "outputs":{n:{"bytes":len(p.read_bytes()),"sha256":H(p.read_bytes())}
            for n,p in [("audit.py",O/"audit.py"),("report.json",O/"report.json")]}
}
mb=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
(O/"manifest.json").write_bytes(mb)
print(json.dumps({"report":report,"manifest_sha256":H(mb)},indent=2))

