from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-toe-extension-substep-velocity-20261111');B=Path('artifacts/crow-v15-hip-roll-staged-toe-extension-20261111')
def records(p):
 with gzip.open(p,'rt') as f:
  for line in f:
   a=line.split()
   if a and a[0]=='step':
    qi=a.index('q'); vi=a.index('v'); yield (int(a[1]),int(a[2])),[float(x) for x in a[qi+1:vi]+a[vi+1:]]
a=iter(records(A/'trace.log.gz'));b=iter(records(B/'trace.log.gz'));maxerr=0.;count=0;first=None
while True:
 try:ka,va=next(a)
 except StopIteration:break
 try:kb,vb=next(b)
 except StopIteration:first=first or ['short baseline',ka];break
 if ka!=kb or len(va)!=len(vb):first=first or [ka,kb];break
 delta=max(abs(x-y) for x,y in zip(va,vb));maxerr=max(maxerr,delta)
 if delta>1e-7 and first is None:first=[ka,delta]
 count+=1
out={'compared_endpoints':count,'maximum_absolute_q_or_v_difference':maxerr,'first_difference_over_1e-7':first,'scope':'End-of-interval state equivalence for matched stationary starts; not a guarantee of identical intermediate contacts.'};(A/'trajectory-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
