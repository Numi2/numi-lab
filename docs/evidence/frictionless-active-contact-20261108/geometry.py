# Exploratory endpoint clearance from the published final quaternion and centre.
# Does not certify the swept path or contact-witness consistency.
import itertools,json,math
from pathlib import Path
p=Path(__file__).parent
for arm in ('baseline','candidate'):
 rows=[]
 for c in json.loads((p/f'{arm}.json').read_text())['cases']:
  q=c['q']; x,y,z,w=q[3:7]; norm=math.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=(t/norm for t in (x,y,z,w))
  # y component of R(q) times each authored box corner.
  yr=(2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w))
  bottom=min(q[1]+sum(a*b for a,b in zip(yr,corn)) for corn in itertools.product((-.2,.2),(-.02,.02),(-.03,.03)))
  rows.append({'angle':c['angle'],'speed':c['speed'],'iterations':c['iterations'],'case':c['case'],'final_clearance_m':bottom})
 (p/f'{arm}-geometry.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(arm,'minimum_final_clearance_m',min(r['final_clearance_m'] for r in rows),'penetrating_1um',sum(r['final_clearance_m']< -1e-6 for r in rows))
