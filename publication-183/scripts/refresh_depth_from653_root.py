#!/usr/bin/env python3
import hashlib,os,sys
from pathlib import Path
w=Path(__file__).resolve().parent;c=w/"combined-43-57-65smooth-109-124-001";candidate=c/"candidate.npz";summary=c/"audit-130/summary.json";surface=c/"audit-159/report.json";query=c/"audit-130/query";out=w/"depth-from653-001";runner=Path("/Users/numi/Work/numi-human-connected-surface-069/D/point-depth-refresh-413/refresh_point_depth_rows.py")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(candidate)=="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af" and not out.exists()
args=[sys.executable,"/tmp/numi-human-geometry-slot.py",sys.executable,str(runner),"--candidate",str(candidate),"--point-summary",str(summary),"--surface-report",str(surface),"--query-dir",str(query),"--output-dir",str(out),"--expected-candidate-sha256",sha(candidate),"--expected-point-summary-sha256",sha(summary),"--expected-surface-report-sha256",sha(surface),"--expected-query-receipt-sha256",sha(query/"query-receipt.json")]
os.execv(sys.executable,args)
