from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-contact-resolution-screen-20261111')
B=Path('artifacts/crow-v15-ankle-near-miss-replay-20261111')
def initial(p):
 with gzip.open(p/'results.txt.gz','rt') as f:return [line for line in f if line.startswith(('initialshape ','initialpose '))]
a,b=initial(A),initial(B);d=[i for i in range(max(len(a),len(b))) if a[i:i+1]!=b[i:i+1]]
r={'records':len(a),'reference_records':len(b),'differences':d};(A/'initial-match.json').write_text(json.dumps(r,indent=2)+'\n');print(r);assert len(a)==len(b)>0 and not d
