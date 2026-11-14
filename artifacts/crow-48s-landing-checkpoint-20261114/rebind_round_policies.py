"""Rebind unchanged policy tensors to the observed round-perch contract."""
from pathlib import Path
p=Path('/tmp/crow-48s-multiview-20261114/rebind_restart_policies.py');source=p.read_text()
needle="err=(r/'artifacts/crow-v15-course-collision-restored-20261114/smoke.stderr').read_text()"
assert source.count(needle)==1
source=source.replace(needle,"import gzip\nerr=gzip.open(a/'round-perch-policy-mismatch/stderr.gz','rt').read()")
source=source.replace("(label+'-rebound.policypack')","(label+'-round-rebound.policypack')")
source=source.replace("b/'libmetalrobo.dylib'","b/'round-perch-observer/libmetalrobo.dylib'")
source=source.replace("a/'policy-rebind.json'","a/'round-perch-policy-rebind.json'")
exec(compile(source,str(p),'exec'),{'__name__':'__main__'})
