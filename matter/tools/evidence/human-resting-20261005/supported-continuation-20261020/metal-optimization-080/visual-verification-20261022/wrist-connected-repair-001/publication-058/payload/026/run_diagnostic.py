from pathlib import Path
import json,subprocess,hashlib,difflib,time
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=E/"native-candidate020-mechanical-source-024";W=E/"skin-material-jacobian-witness-154-zero-trial";N=E/"native-candidate020-failure-witness-026"
N.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(W/"build-004/candidate.metallib")=="76819990f8b17e27621b2e463ff953baa04f05f1bd0aaa94d27f51fbbac0979b"
assert json.loads((W/"coupon-build-004/coupon-receipt.json").read_text())["status"]=="PASS"
base=E/"native-app-main-header-integration-151/snapshot-005"
overlay=W/"source-overlay"
checked={}
for p in overlay.rglob("*"):
 if p.is_file():
  rel=p.relative_to(overlay)
  if str(rel) in ["matter/src/metal/fem.metalinc","matter/src/metal/fem_backtracking_failure_witness.metalinc"]:continue
  ref=base/rel;assert ref.is_file() and p.read_bytes()==ref.read_bytes(),str(rel)
  checked[str(rel)]=sha(p)
(N/"fem-diagnostic-only.diff").write_text("".join(difflib.unified_diff((R/"matter/src/metal/fem.metalinc").read_text().splitlines(True),(overlay/"matter/src/metal/fem.metalinc").read_text().splitlines(True),fromfile="base/fem.metalinc",tofile="witness154/fem.metalinc")))
(N/"compatibility.json").write_text(json.dumps(dict(base_fem_sha256=sha(R/"matter/src/metal/fem.metalinc"),overlay_fem_sha256=sha(overlay/"matter/src/metal/fem.metalinc"),identical_snapshot_files=checked,coupon_receipt_sha256=sha(W/"coupon-build-004/coupon-receipt.json"),scope="Diagnostic shader only; same physical equations, limits, alpha selection and runtime ABI; unsupported release diagnostic, not resting qualification"),indent=2))
cmd=json.loads((D/"native-1ms-command.json").read_text())
cmd[cmd.index("--base-manifest")+1]=str(D/"native-1ms-001/launch-manifest.snapshot.json")
cmd[cmd.index("--run-dir")+1]=str(N/"run-001")
cmd[cmd.index("--metallib")+1]=str(W/"build-004/candidate.metallib")
cmd+=["--source-root",str(overlay),"--source-root",str(W/"manifest.json"),"--env","NM_FEM_NEWTON_TRACE_ROOT=0"]
(N/"command.json").write_text(json.dumps(cmd,indent=2))
p=subprocess.run(cmd,cwd=R)
(N/"runner-exit.json").write_text(json.dumps(dict(returncode=p.returncode)))
raise SystemExit(p.returncode)
