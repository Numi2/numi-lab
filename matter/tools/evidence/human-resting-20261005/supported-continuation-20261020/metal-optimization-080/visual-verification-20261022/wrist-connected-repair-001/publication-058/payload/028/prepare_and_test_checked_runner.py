from pathlib import Path
import hashlib,json,difflib,subprocess,sys,tempfile,os
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/"native-candidate020-bone-witness-028";old=E/"native-candidate020-mechanical-source-024/run_native_contact.py"
s=old.read_text();needle='receipt=dict(returncode=rc,error=error,wall_seconds=time.monotonic()-t,'
assert s.count(needle)==1
check='''# An intentional source-only stop is not evidence that an optional capture
# was implemented. Require each requested initialization diagnostic output.
required_outputs = {
 "NUMI_HUMAN_SKIN_REFERENCE_DUMP": "registered-skin.csv",
 "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP": "registered-skin-poses.json",
 "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP": "tissue-mass-reference.txt",
 "NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV": "initial-skin-bone-contact.csv",
 "NUMI_HUMAN_INITIAL_SKIN_BONE_FULLBANK_COMPARISON_CSV": "initial-skin-bone-fullbank-comparison.csv",
 "NUMI_HUMAN_INITIAL_TET_WITNESS_JSON": "initial-tet.json",
 "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": "initial-skin-fem-nodes.csv",
}
missing_outputs = [dict(environment_key=key, path=str(N/name))
 for key,name in required_outputs.items()
 if env.get(key) and (not (N/name).is_file() or (N/name).stat().st_size == 0)]
capture_complete = not missing_outputs
intentional = intentional and capture_complete
'''
s=s.replace(needle,check+needle).replace('physical_qualification=False,source_only_marker_present=intentional)','physical_qualification=False,source_only_marker_present=marker in out+err,\n requested_diagnostics_complete=capture_complete,missing_requested_outputs=missing_outputs)')
s=s.replace('raise SystemExit(0 if intentional or (rc==0 and error is None) else 1)','raise SystemExit(0 if capture_complete and (intentional or (rc==0 and error is None)) else 1)')
p=D/"run_native_contact_checked.py";p.write_text(s)
(D/"diagnostic-output-completeness.diff").write_text("".join(difflib.unified_diff(old.read_text().splitlines(True),s.splitlines(True),fromfile="024/run_native_contact.py",tofile="028/run_native_contact_checked.py")))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
with tempfile.TemporaryDirectory(prefix="numi-capture-receipt-test-") as td:
 t=Path(td);stub=t/"source-only-fixture";stub.write_text("#!/usr/bin/python3\nimport os,sys,pathlib\nif os.getenv('TEST_WRITE_WITNESS')=='1':pathlib.Path(os.environ['NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV']).write_text('sample,node\\n1,2\\n')\nprint('native tangent source prepared; source-only initialization intentionally not admitted')\nsys.exit(1)\n");stub.chmod(0o755)
 dummy=t/"dummy-runtime";dummy.write_bytes(b"fixture only, no physical runtime")
 for name,request,write,expected in [("requested-missing",True,False,1),("requested-present",True,True,0),("not-requested",False,False,0)]:
  base=t/(name+".json");env={}
  if request:env["NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV"]=str(t/"old-output.csv")
  if write:env["TEST_WRITE_WITNESS"]="1"
  base.write_text(json.dumps(dict(argv=[str(stub),"a","b","c",str(t/"old-output")],environment=env,cwd=str(E.parents[7]))))
  # Use a known repository cwd independent of deep evidence nesting.
  x=json.loads(base.read_text());x["cwd"]="/Users/n/numi-human-aortic-contact-opt-20261019";base.write_text(json.dumps(x))
  cmd=[sys.executable,str(p),"--base-manifest",str(base),"--app",str(stub),"--runtime",str(dummy),"--metallib",str(dummy),"--run-dir",str(t/name),"--mode","source-preflight"]
  z=subprocess.run(cmd,capture_output=True,text=True);assert z.returncode==expected,(name,z.stdout,z.stderr)
  rec=json.load(open(t/name/"execution-receipt.json"))
  assert rec["requested_diagnostics_complete"]==(not request or write)
  assert rec["physical_qualification"] is False
  results.append(dict(case=name,returncode=z.returncode,receipt=rec))
(D/"checked-runner-tests.json").write_text(json.dumps(dict(status="PASS",scope="Synthetic process/receipt tests only; no Metal or human simulation",source_sha256=sha(p),cases=results),indent=2)+"\n")
(D/"028-missing-capture-analysis.json").write_text(json.dumps(dict(status="SOURCE_ONLY_STOP_WITHOUT_REQUESTED_CAPTURE",source_preflight_pass_does_not_validate_capture=True,cause="Current 024/163 app does not implement the older initial bone-contact witness environment hook. The native runner previously did not require requested diagnostic files.",source_run=str(D/"source-preflight-001"),capture_file_missing=True,checked_runner_sha256=sha(p),physical_qualification=False),indent=2)+"\n")
print(json.dumps(dict(status="PASS",runner=str(p),sha256=sha(p),tests=len(results))))
