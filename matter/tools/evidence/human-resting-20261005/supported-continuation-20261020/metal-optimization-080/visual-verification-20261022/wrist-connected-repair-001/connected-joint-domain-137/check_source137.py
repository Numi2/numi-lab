from pathlib import Path
import json,subprocess,hashlib,sys,numpy as np
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019"); E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"; D=E/"connected-joint-domain-137"
C=Path("/tmp/numi-skin-reference137"); sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=C,text=True,capture_output=True)
(D/"tests.log").write_text(p.stdout+p.stderr);assert p.returncode==0
sys.path.insert(0,str(C/"tools"));from human_skin_fat_reference import load_reference
h=json.loads((E/"connected-joint-integration-134/d36-reference-table-check.json").read_text())["input_hashes"]
a=load_reference(E/"wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001/generated",h)
r=dict(status="PASS",test_returncode=p.returncode,source_hashes={str(q.relative_to(C)):sha(q) for q in [C/"tools/human_skin_fat_reference.py",C/"tests/test_human_skin_fat_reference.py"]},d36_input_hashes=h,surface_rows=len(a.outer),faces=len(a.faces),tets=len(a.tetrahedra),materials=np.unique(a.materials).tolist(),scope="typed source identity/overflow regression; no physical or anatomical qualification")
(D/"source-validation.json").write_text(json.dumps(r,indent=2));print(json.dumps(r))
