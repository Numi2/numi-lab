from pathlib import Path
import gzip,json
R=Path.cwd();A=R/'artifacts/crow-v15-sustained-foot-grip-20261110';C=R/'artifacts/crow-v15-launch-candidate-route-20261110'
def states(path):
 d={}
 for line in gzip.open(path,'rt'):
  if line.startswith('CROW_TASK_STATE '):
   s=line.split();t=int(s[2]);v=int(s[3]);d.setdefault(t,{'front_left':v&63,'front_right':(v>>6)&63,'rear_left':(v>>13)&63,'rear_right':(v>>19)&63,'foot_left':float(s[5]),'foot_right':float(s[6])})
 return d
p=states(C/'diagnostic-reserved-final-j-candidate-original.stderr.gz');q=states(A/'diagnostic-reserved-final-j-sustained-original.stderr.gz');times=(1730,1740,1750,1760,1770,1780,1790,1800);out={t:{'baseline':p.get(t),'sustained':q.get(t)} for t in times};(A/'counter-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
