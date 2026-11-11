from pathlib import Path
import json
A=Path('artifacts/crow-v15-landing-root-balance-latch-screen-20261111')
a=json.loads((A/'analysis.json').read_text());v=json.loads((A/'toe-velocity-analysis.json').read_text());rows=[]
for x,w in zip(a['lanes'],v['lanes']):
 contact=x['first_scene_contact'];step=contact['step'] if contact else None
 hits=w['first_episode_hits_after_ground_handoff'];before=[h for h in hits if step is None or h[0]<step];after=[h for h in hits if step is not None and h[0]>=step]
 rows.append({'lane':x['lane'],'first_scene_contact_step':step,'first_episode_end_step':x['first_episode_end_step'],'hits_before_contact':before,'hits_during_or_after_contact':after})
r={'rows':rows,'limitations':'The split uses the first observed scene contact at substep resolution, but reports hits from the same control interval as contact in the after group. It does not rule out below-cap abnormal toe motion, ground contacts, or failures in other starts and layouts.'}
(A/'toe-contact-timing-analysis.json').write_text(json.dumps(r,indent=2)+'\n')
print([(x['lane'],len(x['hits_before_contact']),len(x['hits_during_or_after_contact'])) for x in rows])
