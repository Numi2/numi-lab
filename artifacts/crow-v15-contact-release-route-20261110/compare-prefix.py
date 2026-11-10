import gzip,json
from pathlib import Path
paths=['artifacts/crow-v15-long-perch-j-20261110/development-j-long-perch.json.gz','artifacts/crow-v15-contact-release-route-20261110/development-j-contact-release.json.gz']
frames=[]
for p in paths:
 with gzip.open(p,'rt') as f:frames.append(json.load(f)['payload']['frames'])
result={'expected_frames':5999,'compared_frames':0,'first_difference':None,'fields':['step','q','v']}
for a,b in zip(*frames):
 if a['step']>=6000:break
 result['compared_frames']+=1
 for k in result['fields']:
  if a[k]!=b[k]:result['first_difference']={'step':a['step'],'field':k};break
 if result['first_difference']:break
result['exact_equal']=result['compared_frames']==result['expected_frames'] and result['first_difference'] is None
Path('artifacts/crow-v15-contact-release-route-20261110/prefix-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
