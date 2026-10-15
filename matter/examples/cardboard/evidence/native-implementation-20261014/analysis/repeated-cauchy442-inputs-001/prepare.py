from pathlib import Path
import hashlib, json, shutil
B=Path('/Users/home/cardboard-implementation-20261014')
A=B/'analysis/repeated-cauchy442-inputs-001'
old=B/'solver/repeated-cauchy-full-coupon-438'
case=B/'solver/repeated-cauchy-full-coupon-442'
assert not case.exists();A.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={
    B/'analysis/native433-cauchy-kernel-adjudication-001/report.json':'native433-report.json',
    B/'analysis/native435-enabled-fallback-adjudication-001/report.json':'native435-report.json',
    B/'analysis/repeated-cauchy-selector-review-137-001/report.md':'selector-review.md'}
for p,name in sources.items():
    shutil.copy2(p,A/name);assert sha(p)==sha(A/name)
(A/'manifest.json').write_text(json.dumps(dict(
    scope='Distinct immutable external input basenames repair native438 wrapper preflight; no physical inputs changed.',
    files={name:dict(source=str(p),sha256=sha(p)) for p,name in sources.items()}),indent=2)+'\n')
cmd=json.loads((B/'native-438-command.json').read_text())
for i,s in enumerate(cmd):
    s=s.replace(str(old),str(case))
    if Path(s) in sources:s=str(A/sources[Path(s)])
    cmd[i]=s
(B/'native-442-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
(A/'prepare.py').write_bytes(Path(__file__).read_bytes())
print('prepared442')
