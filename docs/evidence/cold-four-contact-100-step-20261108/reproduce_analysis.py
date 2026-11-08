"""Reconstruct and verify the retained analysis without a GPU."""
import gzip, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
expected={}
for line in (root/"raw-output.sha256").read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    expected[Path(name).name]=digest
with tempfile.TemporaryDirectory(prefix="numi-cold-four-analysis-") as tmp:
    work=Path(tmp)
    for name in ("warm.json","cold.json","friction-warm.json","friction-cold.json"):
        payload=gzip.decompress((root/(name+".gz")).read_bytes())
        if hashlib.sha256(payload).hexdigest()!=expected[name]:
            raise SystemExit("raw output hash mismatch: "+name)
        (work/name).write_bytes(payload)
    for name in ("analyze.py","friction_check.py","verify_friction.py","lever_audit.py","warm.exit","cold.exit"):
        shutil.copyfile(root/name,work/name)
    subprocess.run([sys.executable,str(work/"analyze.py")],check=True,stdout=subprocess.DEVNULL)
    for arm in ("warm","cold"):
        check=subprocess.check_output([sys.executable,str(work/"friction_check.py"),str(work/("friction-"+arm+".json"))])
        (work/("friction-"+arm+"-check.json")).write_bytes(check)
    subprocess.run([sys.executable,str(work/"verify_friction.py")],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(work/"lever_audit.py")],check=True,stdout=subprocess.DEVNULL)
    for name in ("analysis.json","friction-warm-check.json","friction-cold-check.json","friction-verification.json","lever-audit.json"):
        if json.loads((work/name).read_text())!=json.loads((root/name).read_text()):
            raise SystemExit("analysis differs: "+name)
print("raw hashes and all five retained analyses match")
