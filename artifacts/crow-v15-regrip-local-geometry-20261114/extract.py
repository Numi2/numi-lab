from pathlib import Path
import gzip
A=Path("artifacts/crow-v15-regrip-local-geometry-20261114");src=Path("artifacts/crow-v15-contact-gated-regrip-20261114/trace.log.gz");rows=[]
with gzip.open(src,"rt") as f:
 for line in f:
  x=line.split()
  if x and x[0]=="step" and int(x[1]) in (0,1) and int(x[2]) in (250,270,290):
   q=x.index("q");v=x.index("v");rows.append(" ".join(x[1:3]+x[q+1:v]+x[v+1:]))
assert len(rows)==6
(A/"input.tsv").write_text("\n".join(rows)+"\n")
