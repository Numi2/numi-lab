from pathlib import Path
import collections,csv,hashlib,json,statistics
p=Path("artifacts/crow-v15-corvid-flight-data-registry-20261112");source=p/"rsfs20160081_si_001.csv"
with source.open(newline="") as f:rows=list(csv.DictReader(f))
columns=("AirSpeed","Weight","Lift","WeightSupport","DragInduced","DragProfile","DragBody","WingSpan","WingArea","TailSpan","TailArea","BodyAngle","SpanRatio","SpanCamber")
summary={}
for column in columns:
 values=[float(r[column]) for r in rows if r["MoultStage"]=="pre-moult" and r[column] not in ("NA","")]
 summary[column]={"count":len(values),"min":min(values),"median":statistics.median(values),"max":max(values)} if values else {"count":0}
out={"source_doi":"10.6084/m9.figshare.4236305.v2","license":"CC BY 4.0","sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"rows":len(rows),"moult_stages":dict(collections.Counter(r["MoultStage"] for r in rows)),"pre_moult_descriptive":summary,"limitation":"Single jackdaw fast gliding, not independent crow powered-flight trials. Repeated wake measurements are correlated, and missing posture entries remain missing."}
(p/"dataset-summary.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
