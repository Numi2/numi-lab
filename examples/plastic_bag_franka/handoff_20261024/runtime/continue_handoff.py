"""Verify real two-pad support, then continue the requested full native scene."""
from pathlib import Path
import json,time,subprocess,csv,sys
W=Path(__file__).resolve().parents[1];O=W.parents[1]/'outputs'/W.name;REMOTE='/Users/n/numi-plastic-bag-franka-handoff-20261024'
def call(a,**kw):return subprocess.run(a,check=True,**kw)
while not ((O/'capture/run_summary.json').exists() and (O/'watch_capture.log').exists() and 'CAPTURE COMPLETE' in (O/'watch_capture.log').read_text()):
 if (O/'capture_contact_watch.json').exists() and json.loads((O/'capture_contact_watch.json').read_text())['intersection_state_count']:raise RuntimeError('Native capture not clear')
 time.sleep(20)
a=json.loads((O/'capture_contact_watch.json').read_text());assert a['intersection_state_count']==0 and a['through_s']==6.5
fs=json.loads((O/'capture/trajectory.json').read_text())['frames'];assert fs[0]['step']==1200 and fs[-1]['step']==1560
assert all(f['grip_contact_nodes_by_robot'][1]>0 for f in fs[-24:]),'Receiving hand did not hold the corner throughout final 0.1 s'
assert fs[-1]['grip_contact_nodes_by_robot'][0]>0
contacts=[x for x in fs[-1]['grip_contacts'] if x['robot']==1];assert contacts and any(x['node']==550 for x in contacts)
forces=[sum(x[k] for x in contacts) for k in ['left_normal_N','right_normal_N']];assert min(forces)>.1
inter=json.loads(next(l.split('INTERROBOT_AUDIT ',1)[1] for l in (O/'interrobot_planning.log').read_text().splitlines() if l.startswith('INTERROBOT_AUDIT ')));assert inter['intersection_state_count']==0 and inter['saved_states']==1921
for n in ['receiver','donor']:assert json.loads((O/(n+'_table_clearance.json')).read_text())['minimum_gap_m']>=-1e-7
(O/'two_hand_support_before_release.json').write_text(json.dumps({'step':1560,'time_s':6.5,'contact_nodes_by_robot':fs[-1]['grip_contact_nodes_by_robot'],'receiver_contacts':contacts,'receiver_per_jaw_normal_N':forces,'receiver_corner_550_captured':True,'last_0p1_s_receiving_contact_continuous':True,'no_housing_crossing_in_native_capture':True,'giver_will_remain_closed_until_s':6.55},indent=2))
# Dedicated native GPU owner; previous capture job is complete before launch.
call(['ssh','macmini',f'cd {REMOTE} && nohup ./build/numi_classic_bag_handoff --empty-bag --second-gripper --output finish --resume capture/latest_checkpoint.bin --steps 1920 --checkpoint-every 120 > finish.log 2>&1 < /dev/null &'])
(O/'STATUS.json').write_text(json.dumps({'status':'native_full_continuation_running','through_s':6.5,'target_s':8,'receiver_pad_support_measured':True},indent=2));print('FULL NATIVE CONTINUATION STARTED; receiver actual forces '+str(forces),flush=True)
