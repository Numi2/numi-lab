from pathlib import Path
import json,gzip,collections,sys
A=Path('artifacts/crow-v15-mirrored-leg-isolation-20261113');data={}
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-mirrored');m=json.loads((p/'inputs.json').read_text());masks=[];frames={};targets={};errors=[]
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=='CROW_SCENE_MASK':masks.append(z[1:])
   elif z[0]=='CROW_WING_FRAME':frames[int(z[1])]=z[2:]
   elif z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);
    if 335<=t<=400:targets[e,t]=list(map(float,z[3:]))
   elif 'OBSERVER_ERROR' in z[0] or z[0]=='CROW_SUBSTEP_ERROR':errors.append(line.strip())
 data[arm]={'inputs':m,'masks':masks,'frames':frames,'targets':targets,'errors':errors}
p,h=data['push'],data['hold'];same_host_and_packs=all(p['inputs'][k]==h['inputs'][k] for k in ('host_sha256','landing_policy_sha256','route_policy_sha256'));same_masks=p['masks']==h['masks'];frames=p['frames']==h['frames'] and set(p['frames'])==set(range(6));axes=[]
for i in (4,5):
 v=p['frames'].get(i,[]);axes.append(list(map(float,v[4:7])) if len(v)>=7 else None)
corrected_axes=axes==[[0.,1.,0.],[0.,1.,0.]];diff=[];missing=[]
for e in range(8):
 for t in range(335,401):
  a=p['targets'].get((e,t));b=h['targets'].get((e,t))
  if a is None or b is None:missing.append((e,t))
  elif a[6:12]!=b[6:12]:diff.append((e,t))
valid=same_host_and_packs and same_masks and frames and corrected_axes and not diff and not missing and not p['errors'] and not h['errors'];out={'valid':valid,'same_host_and_packs':same_host_and_packs,'same_masks':same_masks,'same_joint_frames':frames,'pronation_axes':axes,'corrected_axes':corrected_axes,'wing_target_differences':diff,'missing_targets':missing,'observer_errors':{'push':p['errors'],'hold':h['errors']},'scope':'Matched corrected-body wing-held controls; not a full launch or biological validation.'};(A/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));sys.exit(0 if valid else 1)
