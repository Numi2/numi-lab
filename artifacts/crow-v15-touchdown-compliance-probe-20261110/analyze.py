from collections import defaultdict
import json
p='artifacts/crow-v15-touchdown-compliance-probe-20261110/'
d=defaultdict(lambda:{'peak':0.,'worst_separation':0.,'first':None,'bad_first_100ms':[], 'complete':False})
for line in open(p+'trace.log'):
 s=line.split()
 if not s:continue
 if s[0]=='case_result':
  k=(int(s[1]),int(s[2]));d[k]['complete']=s[3]=='completed';d[k]['status']=' '.join(s[3:])
 elif s[0]=='trace':
  k=(int(s[1]),int(s[2]));d[k]['peak']=max(d[k]['peak'],float(s[4]))
 elif s[0]=='contact':
  k=(int(s[1]),int(s[2]));t=int(s[3]);name=s[4];imp=float(s[5]);sep=float(s[6]);
  if imp>1e-8:
   if d[k]['first'] is None:d[k]['first']={'step':t,'name':name}
   d[k]['worst_separation']=min(d[k]['worst_separation'],sep)
   if t<20 and '_toe' not in name and '_foot' not in name:d[k]['bad_first_100ms'].append({'step':t,'name':name,'impulse':imp})
res={f'{cs}/{arm}':r for (cs,arm),r in sorted(d.items())}
for arm in range(4):
 print('arm',arm, 'peak',[round(d[cs,arm]['peak'],6) for cs in range(4)],'first',[d[cs,arm]['first'] for cs in range(4)],'worst',[d[cs,arm]['worst_separation'] for cs in range(4)],'bad',[len(d[cs,arm]['bad_first_100ms']) for cs in range(4)], 'complete',[d[cs,arm]['complete'] for cs in range(4)])
 if arm: print('  reductions',[round(1-d[cs,arm]['peak']/d[cs,0]['peak'],3) if d[cs,0]['peak'] else None for cs in range(4)])
json.dump(res,open(p+'analysis.json','w'),indent=2)
