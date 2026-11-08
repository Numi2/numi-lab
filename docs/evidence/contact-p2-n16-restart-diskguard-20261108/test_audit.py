"""Test the pre-analysis audit on disposable synthetic copies only."""
import hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
source = Path(__file__).resolve().parent
root = source.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix="numi-contact-audit-test-") as tmp:
    r = Path(tmp)
    target = r / "docs/evidence/contact-p2-n16-restart-diskguard-20261108"
    target.mkdir(parents=True)
    relative = ["tests/contact/fem_p2_full_restart.py", "tests/contact/fem_p2_dynamics.py", "docs/evidence/contact-p2-n8-order18-20261108/fine.json", "docs/evidence/contact-p2-n8-restart-full-20261108/analysis.json", "docs/evidence/contact-p2-n8-restart-full-20261108/fine.json"]
    for path in relative:
        destination = r / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, destination)
    for name in ("audit_before_analysis.py", "analyze.py", "coarse.json", "coarse-manifest.json"):
        shutil.copyfile(source / name, target / name)
    coarse = json.loads((target / "coarse.json").read_text())
    fake = dict(coarse, variant="fine")
    fine = target / "fine.json"
    fine.write_text(json.dumps(fake))
    manifest = json.loads((target / "coarse-manifest.json").read_text())
    manifest.update(variant="fine", output_sha256=sha(fine))
    mf = target / "fine-manifest.json"
    mf.write_text(json.dumps(manifest))
    def check(label, want):
        result = subprocess.run([sys.executable, str(target / "audit_before_analysis.py")], capture_output=True, text=True)
        if (result.returncode == 0) != want:
            raise SystemExit(f"{label}: unexpected exit {result.returncode}: {result.stderr} {result.stdout}")
        print(label + (": expected pass observed" if want else ": expected rejection observed"))
    check("synthetic valid", True)
    manifest["exit"] = 1; mf.write_text(json.dumps(manifest)); check("failed process", False)
    manifest["exit"] = 0; manifest["output_sha256"] = "0" * 64; mf.write_text(json.dumps(manifest)); check("changed output hash", False)
    manifest["output_sha256"] = sha(fine); manifest["source_sha256"] = "0" * 64; mf.write_text(json.dumps(manifest)); check("changed input hash", False)
    manifest["source_sha256"] = sha(r / relative[0])
    def set_fake(value):
        fine.write_text(json.dumps(value)); manifest["output_sha256"] = sha(fine); mf.write_text(json.dumps(manifest))
    broken = dict(fake, trace=fake["trace"][:-1]); set_fake(broken); check("truncated trace", False)
    broken = dict(fake, force_bins_N=[float("nan")] + fake["force_bins_N"][1:]); set_fake(broken); check("nonfinite force", False)
    broken = dict(fake, force_bins_N=[0.0]*10); set_fake(broken); check("zero force normalization", False)
    broken = dict(fake, trace=[list(row) for row in fake["trace"]]); broken["trace"][2][0] = broken["trace"][1][0]; set_fake(broken); check("nonmonotone time", False)
