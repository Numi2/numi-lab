from pathlib import Path
from collections import defaultdict
import json
A=Path('artifacts/crow-v15-grip-search-20261109');lines=(A/'perch-contact-identity-results.txt').read_text().splitlines();old=(A/'perch-control-results.txt').read_text().splitlines();assert [l for l in lines if l.startswith(('trace ','arm_result '))]==old
C=defaultdict(float);T={};res=[]
for l in lines:
 a=l.split()
 if a[0]=='contact':
  arm,step=int(a[1]),int(a[2]);name=a[3];nx=float(a[4]);imp=float(a[6]);side='left' if '_left_' in name else 'right';digit='rear' if '_rear_' in name else 'front';direction='positive' if nx>.5 else 'negative' if nx<-.5 else 'other';C[arm,step,side,digit,direction]+=imp
 if a[0]=='trace':T[int(a[1]),int(a[2])]=list(map(float,a[3:]))
 if a[0]=='arm_result':res.append(l)
for side in ('left','right'):
 hold=[step for step in range(67,79) if C[0,step,side,'front','negative']>.00005 and C[0,step,side,'rear','positive']>.00005]
 lost=next((step for step in range(79,150) if C[0,step,side,'front','negative']<=.00005 or C[0,step,side,'rear','positive']<=.00005),None)
 print(side,'independently opposing steps',hold,'first release loss',lost)
 report={'side':side,'opposing_hold_steps':hold,'first_release_loss_step':lost};res.append(report)
(A/'perch-contact-identity-analysis.json').write_text(json.dumps({'trace_exact_match':True,'results':res,'scope':'isolated estimated toe body, exposed perch; no disturbance, integration, anatomy or full sequence qualification'},indent=2)+'\n')
