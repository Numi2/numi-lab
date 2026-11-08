from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-active-head-navigation-20261108';B=R/'build/crow-active-head-navigation-admission-01'
B.mkdir(exist_ok=False)
binary=R/'build/bin/metalrobo_task_rollout'
base=[str(binary),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15','--envs','1','--steps','8','--repeats','1','--chunk','1','--physics-substeps','32','--seed','20261108','--no-scheduled-resets']
legacy='assets/crow_navigation_course/crow-navigation.sensor-fast.visual-observation.json'
head='assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json'
policy='environments/american-crow/action-holds-v1/policies/deployment97.policypack'
cases=[('legacy_camera',base+['--zero-actions','--visual-observation-config',legacy]),
       ('head_zero',base+['--zero-actions','--visual-observation-config',head,'--crow-replay-pack',str(B/'replay.json')]),
       ('feeding_policy',base+['--policy-pack',policy,'--visual-observation-config',head])]
records=[]
for name,cmd in cases:
    start=time.monotonic()
    with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
        try: code=subprocess.run(cmd,cwd=R,stdout=out,stderr=err,timeout=180).returncode
        except subprocess.TimeoutExpired:code=None
    records.append(dict(name=name,argv=cmd,returncode=code,seconds=time.monotonic()-start))
    (A/'execution.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
    print(json.dumps(records[-1]),flush=True)
out=(B/'head_zero.stdout').read_text();positive=None
try: positive=json.loads(out.strip().splitlines()[-1])
except (ValueError,IndexError): pass
legacy_message=(B/'legacy_camera.stderr').read_text()
policy_message=(B/'feeding_policy.stderr').read_text()
replay=None
if (B/'replay.json').exists():
    replay=json.loads((B/'replay.json').read_text())['payload']
checks={
 'legacy_torso_camera_rejected':records[0]['returncode']!=0 and 'head-mounted' in legacy_message,
 'new_task_accepted_8_zero_action_steps':bool(positive and records[1]['returncode']==0 and positive['action_count']==18 and positive['failed_environment_steps']==0 and positive['visual_observation'] and positive['action_source']=='zero' and replay and replay['frame_count']==8),
 'feeding_policy_rejected':records[2]['returncode']!=0 and ('fingerprint' in policy_message.lower() or 'dimension' in policy_message.lower() or 'shape' in policy_message.lower() or 'layout' in policy_message.lower()),
}
result=dict(schema='crow-active-head-navigation-admission-v1',checks=checks,all_checks_passed=all(checks.values()),
  positive=positive,legacy_error=legacy_message[-2500:],policy_error=policy_message[-2500:],
  replay_sha256=hashlib.sha256((B/'replay.json').read_bytes()).hexdigest() if replay else None,
  scope='Eight zero-action steps only; not learned navigation, vision qualification, or a held-out outcome.')
(A/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(checks,sort_keys=True),flush=True)
