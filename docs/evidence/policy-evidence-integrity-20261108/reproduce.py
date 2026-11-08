import json,sys
sys.path.insert(0,sys.argv[1]+"/python")
from metalrobo.policy_selection import compare_evidence
base={"task":"grasp","world_source":"world_pack","termination_count_by_environment":[0,0],"termination_count":0,"termination_reason_counts":{},"physics_error_count_by_environment":[0,0],"failed_environment_steps":0,"mean_task_reward":0.1,"mean_reward":0.1,"mean_tracking_score":0.0}
for name in ("missing_total","missing_failed_steps"):
 candidate={**base,"mean_task_reward":0.9,"mean_reward":0.9}
 if name=="missing_total":candidate["termination_count_by_environment"]=[1,0];del candidate["termination_count"]
 else:del candidate["failed_environment_steps"]
 print(json.dumps({"case":name,"decision":compare_evidence(base,candidate)},sort_keys=True))
