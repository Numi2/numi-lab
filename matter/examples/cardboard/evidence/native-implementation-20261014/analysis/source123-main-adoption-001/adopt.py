from pathlib import Path
import hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-two-stage-material-merit-123';P=B/'snapshots/source-valid-summary-tet858-121';O=B/'analysis/source123-main-adoption-001';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((P/'frozen-source.json').read_text())['source_sha256'];b=json.loads((S/'frozen-source.json').read_text())['source_sha256'];rows=[]
for p,h in b.items():
 assert sha(S/p)==h,p
 if a.get(p)==h:continue
 assert sha(R/p)==a[p],p
 rows.append(dict(path=p,base_sha256=a[p],adopted_sha256=h))
assert len(rows)==5,rows
for row in rows:shutil.copy2(S/row['path'],R/row['path']);assert sha(R/row['path'])==row['adopted_sha256']
report=dict(schema='numi.cardboard.source123-main-adoption.v1',source_snapshot=str(S),changes=rows,build_sha256=sha(R/'build-cardboard-provenance-123a/source-build.json'),qualification='Native point407/408/411 and3072sample410regression. Fullcoupon409ongoing; no completecoupon/convergence orfullbox qualification. ExactpointnativeFP64comparison and read-onlyVTCCD405406historyparity sealed separately.',physical_validation=False)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'adopt.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n');print(json.dumps(rows,indent=2))
